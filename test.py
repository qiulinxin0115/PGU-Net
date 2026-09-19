"""Full-image enhancement and optional paired RGB metrics."""
import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import ImageNormalDataset
from utils import (load_checkpoint, model_from_checkpoint, select_device,
                   move_batch, save_rgb, metrics, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--low-dir", required=True)
    parser.add_argument("--normal-dir", required=True)
    parser.add_argument("--high-dir", help="Optional paired normal-light references.")
    parser.add_argument("--out-dir", required=True)
    for name in ("low", "high", "normal"):
        parser.add_argument("--" + name + "-prefix", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 0 or args.cpu_threads < 1:
        parser.error("Invalid worker/thread count.")
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    model = model_from_checkpoint(checkpoint).to(device).eval()
    dataset = ImageNormalDataset(args.low_dir, args.normal_dir, args.high_dir,
                                 low_prefix=args.low_prefix, high_prefix=args.high_prefix,
                                 normal_prefix=args.normal_prefix)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory to avoid mixing results.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "enhanced").mkdir()
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            prediction, _ = model(batch["low"], batch["normal"])
            prediction = prediction.clamp(0, 1)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite prediction.")
            name = batch["name"][0]
            save_rgb(out / "enhanced" / (name + ".png"), prediction)
            record = {"name": name}
            if "high" in batch:
                record.update(metrics(prediction, batch["high"]))
            records.append(record)
            print("Processed", name, flush=True)
    summary = {"samples": len(records), "checkpoint_epoch": checkpoint["epoch"],
               "metric_protocol": "float RGB [0,1], no border crop; SSIM Gaussian11 sigma1.5 zero padding"}
    if args.high_dir:
        summary.update({k: sum(r[k] for r in records) / len(records)
                        for k in ("psnr", "ssim", "mae")})
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(out / "evaluation.json", {"arguments": vars(args), "summary": summary,
                                         "samples": records})
    print(summary)


if __name__ == "__main__":
    main()
