"""Train the released PGU-Net core using paired images and pretrained PBSIR."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import ImageNormalDataset
from models import PGUNet, PBSIRModule
from losses import PGUNetLoss
from utils import (FORMAT, seed_everything, seed_worker, select_device, move_batch,
                   metrics, load_checkpoint, model_from_checkpoint, save_checkpoint,
                   write_json, rng_state, restore_rng)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("low", "high", "normal"):
        parser.add_argument("--train-" + name, required=True)
        parser.add_argument("--val-" + name)
        parser.add_argument("--" + name + "-prefix", default="")
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--pbsir-checkpoint",
                                help="Compatible pretrained mirror_net/light_predictor checkpoint.")
    initialization.add_argument("--resume", help="Resume a checkpoint saved by this train.py.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256, help="0 for original image size.")
    parser.add_argument("--augment", action="store_true",
                        help="Paired horizontal/vertical flips with normal-vector sign changes.")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--light-noise-sigma", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="Optional planned stop, preserving the full --epochs LR schedule.")
    args = parser.parse_args()
    positive = (args.epochs, args.batch_size, args.stages, args.val_every, args.cpu_threads)
    if any(v < 1 for v in positive) or args.num_workers < 0:
        parser.error("Epoch/batch/stage/interval/thread counts must be positive; workers >= 0.")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        parser.error("Require 0 <= min-lr <= lr and lr > 0.")
    if args.grad_clip <= 0 or args.light_noise_sigma < 0:
        parser.error("Require grad-clip > 0 and light-noise-sigma >= 0.")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        parser.error("stop-after-epoch must be between 0 and epochs.")
    validation = (args.val_low, args.val_high, args.val_normal)
    if any(validation) and not all(validation):
        parser.error("Supply all three validation folders, or none.")
    if args.crop_size == 0 and args.batch_size != 1:
        parser.error("Use batch-size 1 for full-resolution training.")
    return args


def make_dataset(args, split):
    return ImageNormalDataset(
        getattr(args, split + "_low"), getattr(args, split + "_normal"),
        getattr(args, split + "_high"),
        crop_size=args.crop_size if split == "train" else 0,
        augment=args.augment if split == "train" else False,
        low_prefix=args.low_prefix, high_prefix=args.high_prefix,
        normal_prefix=args.normal_prefix)


def compute_losses(model, teacher, criterion, batch, sigma):
    output, history = model(batch["low"], batch["normal"])
    low_parameters = {k: history[k][-1] for k in ("ks", "n", "light")}
    with torch.no_grad():
        high_parameters = teacher(batch["high"], batch["normal"])
    perturbed_light = model.pbsir.perturb_light(low_parameters["light"].detach(), sigma)
    robust_parameters = model.pbsir.mirror_with_light(history["I_int"][-1], perturbed_light)
    return criterion(output, batch["high"], batch["low"],
                     history["D"][-1], history["S"][-1], history["M"][-1],
                     history["M_pbsir"][-1], low_parameters, high_parameters, robust_parameters)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        output, _ = model(batch["low"], batch["normal"])
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite validation output.")
        rows.append(metrics(output, batch["high"]))
    return {k: sum(row[k] for row in rows) / len(rows) for k in rows[0]}


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = select_device(args.device)
    resume = load_checkpoint(args.resume) if args.resume else None
    # Preserve optimizer/scheduler/data semantics when resuming.
    resume_keys = ("epochs", "batch_size", "lr", "min_lr", "stages", "crop_size",
                   "augment", "grad_clip", "light_noise_sigma", "seed", "num_workers",
                   "low_prefix", "high_prefix", "normal_prefix")
    if resume:
        changed = [k for k in resume_keys if vars(args)[k] != resume["training_config"][k]]
        if changed:
            raise ValueError("Resume with the original settings: " + ", ".join(changed))
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("Use an empty output directory for a new training run.")
    out.mkdir(parents=True, exist_ok=True)
    train_set = make_dataset(args, "train")
    val_set = make_dataset(args, "val") if args.val_low else None
    if val_set is not None:
        train_paths = {p[1].resolve() for p in train_set.samples}
        if train_paths.intersection(p[1].resolve() for p in val_set.samples):
            raise ValueError("Training and validation must not use the same low-light files.")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, worker_init_fn=seed_worker,
                        generator=generator)
    val_loader = (DataLoader(val_set, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, worker_init_fn=seed_worker)
                  if val_set is not None else None)
    model_config = resume["model_config"] if resume else {"stages": args.stages}
    model = (model_from_checkpoint(resume) if resume else PGUNet(
        pbsir_checkpoint=args.pbsir_checkpoint, **model_config)).to(device)
    teacher = PBSIRModule().to(device)
    if resume:
        teacher.load_state_dict(resume["teacher"], strict=True)
    else:
        teacher.load_state_dict(model.pbsir.state_dict(), strict=True)
    teacher.set_trainable(False)
    teacher.eval()
    model.set_pbsir_trainable(True)
    criterion = PGUNetLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)
    start_epoch, best_psnr = 1, None
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch = resume["epoch"] + 1
        best_psnr = resume["best_psnr"]
        restore_rng(resume["rng"], generator)
    final_epoch = args.stop_after_epoch or args.epochs
    if final_epoch < start_epoch:
        raise ValueError("No epochs left to run; check --epochs and --stop-after-epoch.")
    write_json(out / ("resume_arguments.json" if resume else "arguments.json"), vars(args))
    print("Device:", device, "| training:", len(train_set),
          "| validation:", len(val_set) if val_set else 0, flush=True)
    print("All loss terms enabled; pretrained PBSIR teacher fixed; joint Adam optimization.", flush=True)
    for epoch in range(start_epoch, final_epoch + 1):
        model.train()
        totals, count = {}, 0
        current_lr = optimizer.param_groups[0]["lr"]
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            terms = compute_losses(model, teacher, criterion, batch, args.light_noise_sigma)
            if not all(torch.isfinite(v).all() for v in terms.values()):
                raise FloatingPointError("Non-finite training loss.")
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                           error_if_nonfinite=True)
            optimizer.step()
            batch_size = batch["low"].shape[0]
            count += batch_size
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
        scheduler.step()
        validation = (validate(model, val_loader, device)
                      if val_loader and (epoch % args.val_every == 0 or epoch == final_epoch)
                      else None)
        improved = validation is not None and (best_psnr is None or validation["psnr"] > best_psnr)
        if improved:
            best_psnr = validation["psnr"]
        record = {"epoch": epoch, "lr": current_lr,
                  "next_lr": optimizer.param_groups[0]["lr"],
                  "train": {k: v / count for k, v in totals.items()},
                  "validation": validation}
        state = {"format": FORMAT, "epoch": epoch, "best_psnr": best_psnr,
                 "model_config": model_config, "training_config": vars(args),
                 "model": model.state_dict(), "teacher": teacher.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": rng_state(generator), "metrics": validation}
        save_checkpoint(out / "latest.pth", state)
        if improved:
            save_checkpoint(out / "best.pth", state)
        import json
        with (out / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print("Epoch {}/{} | loss {:.6f} | validation {}".format(
            epoch, args.epochs, record["train"]["total"], validation), flush=True)
    print("Saved:", out / "latest.pth", flush=True)


if __name__ == "__main__":
    main()

