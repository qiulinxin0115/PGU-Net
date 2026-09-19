# PGU-Net

**Physics-Guided Unfolding with Specular Inverse Rendering for Low-Light Image Enhancement**

This repository provides the PGU-Net core network, loss functions, joint enhancement training code, and inference/evaluation code.

## Contents

```text
PGU-Net/
├── train.py                 # Joint enhancement training and checkpoint resume
├── test.py                  # Full-image inference and optional paired metrics
├── data.py                  # Paired images, normal maps and spatial transforms
├── utils.py                 # Checkpoints, metrics, image output and random states
├── models/                  # D-Net, S-Net, PBSIR and learned specular modulation
├── losses/                  # Reconstruction, component and parameter losses
├── results/LOLv1/           # 15 precomputed enhancement outputs
├── requirements.txt
└── README.md
```

The default model shares weights across four unfolding stages. It takes low-light RGB and a fixed surface-normal map as input, and predicts diffuse D, compensation S, and specular M components. The reconstructed image is D + M + S.

## Installation

Tested with Python 3.8.20 and PyTorch 2.3.1. Install a PyTorch build suitable for your CUDA platform, then:

```bash
pip install -r requirements.txt
```

No external configuration file is required. Run `python train.py --help` or `python test.py --help` for all arguments.


## Data layout

Use one dataset's training split per run. Keep any validation split separate from the training samples; use the held-out test split for final evaluation.

```text
data/LOLv1/
├── train/
│   ├── low/1.png
│   ├── high/1.png
│   └── normal/1.npy
├── val/                     # Optional validation split
│   ├── low/2.png
│   ├── high/2.png
│   └── normal/2.npy
└── test/
    ├── low/3.png
    ├── high/3.png
    └── normal/3.npy
```

Images are read as RGB in [0,1].


## Training

The main defaults follow the settings: Adam, 200 epochs, batch size 1, learning rate 1e-4 decaying to 1e-5 with cosine annealing, and four unfolding stages. All trainable model branches are optimized jointly. The PBSIR teacher is initialized from the supplied pretrained weights and kept fixed for the parameter-invariance target. Light perturbation supplies the robustness term.

The script additionally exposes crop size (default 256), optional flips, gradient clipping (default 5), light-noise standard deviation (default 0.05), and random seed (default 42).

```bash
python train.py --train-low data/LOLv1/train/low --train-high data/LOLv1/train/high --train-normal data/LOLv1/train/normal --pbsir-checkpoint weights/pbsir_pretrained.pth --out-dir runs/lolv1 --device cuda
```

Random crops apply identically to images and normals. `--augment` enables paired horizontal/vertical flips and corrects the corresponding normal-vector signs. Use `--crop-size 0` with batch size 1 for full-resolution training. Images smaller than the requested crop are rejected rather than silently resized.

Each epoch saves `latest.pth`. When validation is supplied, improved validation PSNR also saves `best.pth`. The run directory contains the resolved CLI arguments and a JSONL log of all loss terms, learning rates and validation metrics.

Resume using the original training settings and paths, replacing `--pbsir-checkpoint` with `--resume`:

```bash
python train.py --train-low data/LOLv1/train/low --train-high data/LOLv1/train/high --train-normal data/LOLv1/train/normal --resume runs/lolv1/latest.pth --out-dir runs/lolv1 --device cuda
```

Checkpoints preserve the model, fixed teacher, optimizer, scheduler, and random states. Keep the original total `--epochs` value when resuming. For a planned early stop without changing that learning-rate schedule, use `--stop-after-epoch`.

## Testing and image enhancement

```bash
python test.py --checkpoint runs/lolv1/latest.pth --low-dir data/LOLv1/test/low --normal-dir data/LOLv1/test/normal --high-dir data/LOLv1/test/high --out-dir outputs/lolv1 --device cuda
```


Inference uses the full input resolution. Metrics are computed on clipped floating-point RGB before PNG quantization, with no border crop or luminance conversion, and averaged equally across images. SSIM uses an 11x11 Gaussian window, sigma 1.5 and zero padding, matching the SSIM implementation in this package. PSNR uses a minimum MSE of 1e-12. 

## Loss and implementation notes

The loss coefficients are L1=1, SSIM=0.1, D-allocation=0.5, D-structure=0.05, S-reference=1.5, S-TV=0.01, M-physics=0.05, M-sparsity=0.0005, parameter-invariance=0.03 and light-robustness=0.01. They are defined in `losses/objectives.py`. 

D/S updates use nine-channel concatenations. S is constrained to a low-frequency compensation with maximum amplitude 0.25. The intermediate reconstruction is clipped before PBSIR; the final reconstruction remains an unclipped sum for L1 training. The default D-Net and PBSIR keep BatchNorm running statistics fixed. The learned P_M module multiplies the physical specular response by a bounded modulation. These implementation details are visible in the source.


## Results

The [LOLv1 results](results/LOLv1) are supplied outputs. 
The [LOLv2 results](results/LOLv2) are supplied outputs. 


## Acknowledgements

The method uses DSINE surface-normal estimates and the LOL dataset. 

