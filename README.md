# LDM from Scratch

[中文文档](README_zh.md)

A minimal, single-file reimplementation of **Latent Diffusion Models** (Rombach et al., CVPR 2022). The official vq-f4 is frozen; an unconditional U-Net is trained on CelebA-HQ 256² latents (64×64×3); samples are drawn with DDIM and decoded.

Official config: [celebahq-ldm-vq-4.yaml](https://github.com/CompVis/latent-diffusion/blob/main/configs/latent-diffusion/celebahq-ldm-vq-4.yaml) (LDM-4). Inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT) style: one script, readable code, easy to learn and extend.

**Paper:**

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## Features

- Frozen official vq-f4 (`encode` returns pre-quant z; `decode` does nearest-codebook lookup)
- Forward diffusion and L_simple (ε-prediction) training
- DDIM sampling: 200-step sub-sequence by default, adjustable η (η=0 deterministic, η=1 DDPM-level variance)
- Latent U-Net with sinusoidal timestep embedding and spatial self-attention
- EMA weights for sampling; fixed noise for training grids
- TensorBoard (loss, speed, sample grids, FID)
- FID against training images (Inception-v3, 2048-d)
- Auto-download CelebA-HQ, or a local ImageFolder

## Project Layout

```
my_LDM/
├── train.py              # training, sampling, FID (all-in-one)
├── requirements.txt
├── docs/
│   ├── LDM-note.html     # paper notes (Chinese)
│   ├── 2112.10752v2.pdf
│   └── figs/
├── models/vq-f4/         # official vq-f4
├── data/                 # CelebA-HQ
└── runs/
```

## Setup

Requires Python 3.11+, NVIDIA GPU recommended (CUDA 12.0+ driver).

```bash
pip install -r requirements.txt
```

```bash
mkdir -p models/vq-f4
wget -O models/vq-f4/vq-f4.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
unzip -o models/vq-f4/vq-f4.zip -d models/vq-f4
```

This yields `models/vq-f4/model.ckpt`.

## Quick Start

### Train

If `--data_dir` is already an ImageFolder, it is used as-is; otherwise CelebA-HQ 256² is downloaded from Hugging Face and exported to `data/celebahq/all/`.

```bash
python train.py --vq_ckpt models/vq-f4/model.ckpt \
  --batch_size 8 --lr 1.6e-5 --max_steps 80000 \
  --ddim_steps 50 --n_samples 8 \
  --sample_every 2000 --ckpt_every 10000
```

`lr = 2e-6 × batch`. Outputs go to `runs/ldm_uncond/`.

### Resume

```bash
python train.py --vq_ckpt models/vq-f4/model.ckpt \
  --resume runs/ldm_uncond/ckpt.pt --out runs/ldm_uncond
```

### Sample

`--sample` uses EMA weights, 200-step DDIM, η=0 by default:

```bash
python train.py --sample --ckpt runs/ldm_uncond/ckpt.pt \
  --vq_ckpt models/vq-f4/model.ckpt --out runs/ldm_uncond
python train.py --sample --ckpt runs/ldm_uncond/ckpt.pt \
  --vq_ckpt models/vq-f4/model.ckpt --eta 1.0
```

Writes `samples_final.png` and `progression.png` (`--progress_every 10`). With `--eta 0` and the same seed, repeated runs are pixel-identical.

### Local ImageFolder

```bash
python train.py --vq_ckpt models/vq-f4/model.ckpt --data_dir my_faces
```

Layout: `root/class_name/*.{jpg,png,webp}`.

### TensorBoard

```bash
tensorboard --logdir runs/ldm_uncond/tb
```

Logs: `train/loss`, `train/ms_per_step`, `samples/grid`, `eval/fid`.

### FID Evaluation

```bash
python train.py --eval_fid --ckpt runs/ldm_uncond/ckpt.pt \
  --vq_ckpt models/vq-f4/model.ckpt --n_fid 5000 --ddim_steps 200
```

During training (`--fid_every 0` disables):

```bash
python train.py --vq_ckpt models/vq-f4/model.ckpt --fid_every 20000 --n_fid 1000
python train.py --vq_ckpt models/vq-f4/model.ckpt --fid_final --n_fid 10000
```

## Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max_steps` | 410000 | Training steps |
| `--batch_size` | 48 | Batch size |
| `--lr` | 9.6e-5 | AdamW (`2e-6 × batch`) |
| `--dim` | 224 | U-Net base channels (~228M) |
| `--T` | 1000 | Diffusion timesteps |
| `--ddim_steps` | 200 | DDIM sub-sequence length |
| `--eta` | 0.0 | 0 = deterministic, 1 = DDPM variance |
| `--sample_every` | 5000 | Save sample grid every N steps |
| `--n_fid` | 10000 | Images for FID |
| `--fid_every` | 0 | FID every N steps (`0` = disable) |

## Dataset

| Item | This repo | Paper |
|------|-----------|-------|
| Dataset | [korexyz/celeba-hq-256x256](https://huggingface.co/datasets/korexyz/celeba-hq-256x256) | taming `CelebAHQTrain` |
| Task | Unconditional generation | Unconditional generation |
| Training data | train+val, ~30k | train split |
| Resolution | 256×256 RGB | 256×256 RGB |
| Latent | 64×64×3 (vq-f4) | same |
| Preprocessing | Resize / CenterCrop, [-1, 1] | resize 256, [-1, 1] |
| Augmentation | Random horizontal flip | Random horizontal flip |
| FID reference | Training images (no flip) | CelebA-HQ train set |

## Results

FID: generated samples vs. training images (Inception-v3, 2048-d). EMA + DDIM. Hardware: RTX 4090D. This repo: batch 8, lr 1.6e-5, 80k steps, ~220 ms/step.

| Metric | This repo | Paper (LDM-4) |
|--------|-----------|---------------|
| **FID ↓** (DDIM, η=0) | **21.9** (5k images, 200 steps, 83.5 min) | **5.11** (200 steps) |
| Training steps | 80k | 410k |
| Batch size | 8 | 48 |
| U-Net | ~228M | ~274M |

The gap is from U-Net structure, training steps, and batch size. Fixed-noise grids:

<p align="center">
  <img src="docs/figs/samples_2000.png" width="48%"/>
  <img src="docs/figs/samples_20000.png" width="48%"/>
</p>
<p align="center"><em>2k / 20k</em></p>

<p align="center">
  <img src="docs/figs/samples_40000.png" width="48%"/>
  <img src="docs/figs/samples_80000.png" width="48%"/>
</p>
<p align="center"><em>40k / 80k</em></p>

## vs. Original Paper

The core algorithm matches CelebA-HQ LDM-4 (frozen vq-f4, ε-prediction, L_simple, sqrt-space linear β, EMA, DDIM). The U-Net module tree differs; official U-Net weights cannot be loaded.

| Item | This repo | Paper (LDM-4) |
|------|-----------|---------------|
| First-stage | official vq-f4, frozen | same |
| U-Net | ~228M; 2 ResBlocks per level, one attention / skip | ~274M `UNetModel`; attention after every ResBlock, concat per up-block |
| Training steps | 80k | 410k |
| Batch size | 8 | 48 |
| Optimizer / lr | AdamW, 1.6×10⁻⁵ | AdamW, 9.6×10⁻⁵ |
| EMA | 0.9999 | 0.9999 (LitEma warmup) |
| Dropout | 0.0 | 0.0 |
| T / β | 1000, `linspace(√0.0015, √0.0195)²` | same |
| DDIM sub-sequence | fractional uniform, η=0 | `range(0,T,T//n)`, η=0 |
| FID reference / scale | this repo's train set, 5k images | CelebA-HQ train; paper reports 5.11 |

See `docs/LDM-note.html` for a detailed paper walkthrough.

## Reference

```bibtex
@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={CVPR},
  year={2022}
}
```
