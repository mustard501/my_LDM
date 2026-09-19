# LDM 复现

基于 Rombach et al. (CVPR 2022) 的 **Latent Diffusion Models** 简化复现：冻结官方 vq-f4，在 CelebA-HQ 256² 的 latent 上训练无条件 U-Net，DDIM 采样后再解码回像素。

对应官方配置：[celebahq-ldm-vq-4.yaml](https://github.com/CompVis/latent-diffusion/blob/main/configs/latent-diffusion/celebahq-ldm-vq-4.yaml)（LDM-4）。

代码风格参考 [nanoGPT](https://github.com/karpathy/nanoGPT)：单文件、可读、便于学习与扩展。

**论文：**

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## 功能

- 冻结官方 vq-f4：256² RGB → 64×64×3 latent（`encode` 为量化前 z；`decode` 做 codebook 最近邻后再解码）
- 前向扩散、L_simple（ε-预测）训练
- DDIM 采样（默认 200 步，η 可调；η=0 确定性）
- 轻量化 latent U-Net（正弦时间嵌入 + 空间自注意力）
- EMA 权重采样；训练中间出图使用固定噪声，便于对比不同 step
- TensorBoard 监控（loss、速度、采样图、FID）
- FID 评测（相对当前训练图像，Inception-v3 2048 维；规模由命令行控制）
- CelebA-HQ 自动下载，或使用本地 ImageFolder

不包含：DDPM 采样、class / text 条件、KL first-stage、官方 U-Net 权重加载。

## 项目结构

```
my_LDM/
├── train.py              # 训练 / 采样（单文件）
├── requirements.txt
├── docs/
│   └── LDM-note.html     # 论文精读笔记
├── models/vq-f4/         # 官方 vq-f4（需手动下载）
├── data/                 # CelebA-HQ 缓存（自动下载）
└── runs/                 # checkpoint、采样图、tensorboard 日志
```

## 环境安装

需要 Python 3.11+，建议使用 NVIDIA GPU（驱动支持 CUDA 12.0+）。

```bash
pip install -r requirements.txt
```

第一阶段权重需单独下载（训练不会自动拉取）：

```bash
mkdir -p models/vq-f4
wget -O models/vq-f4/vq-f4.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
unzip -o models/vq-f4/vq-f4.zip -d models/vq-f4
```

解压后应有 `models/vq-f4/model.ckpt`。

## 使用示例

### 训练

`--data_dir` 已是 ImageFolder 则直接读；否则从 Hugging Face 下载 CelebA-HQ 256²，导出到 `data/celebahq/all/`。

论文默认（batch 48、410k step）单卡 4090 容易 OOM / 过久。单卡完整跑通建议：

```bash
python train.py --vq_ckpt models/vq-f4/model.ckpt \
  --batch_size 8 --lr 1.6e-5 --max_steps 80000 \
  --ddim_steps 50 --n_samples 8 \
  --sample_every 2000 --ckpt_every 10000
```

`lr` 按官方比例 `2e-6 × batch`。checkpoint 与 `samples_{step}.png` 写在 `runs/ldm_uncond/`。

### 断点续训

```bash
python train.py --vq_ckpt models/vq-f4/model.ckpt \
  --resume runs/ldm_uncond/ckpt.pt --out runs/ldm_uncond
```

### 采样

`--sample` 只走 DDIM（默认 200 步、η=0），使用 checkpoint 里的 EMA 权重：

```bash
python train.py --sample --ckpt runs/ldm_uncond/ckpt.pt \
  --vq_ckpt models/vq-f4/model.ckpt --out runs/ldm_uncond
python train.py --sample --ckpt runs/ldm_uncond/ckpt.pt \
  --vq_ckpt models/vq-f4/model.ckpt --eta 1.0
```

生成 `samples_final.png` 和 `progression.png`（各行是轨迹上解码后的 x0_hat，由粗到细；`--progress_every 10` 控制间隔）。

### 本地 ImageFolder

目录需为 `root/class_name/*.{jpg,png,webp}`，指向该根目录则不再下载：

```bash
python train.py --vq_ckpt models/vq-f4/model.ckpt --data_dir my_faces
```

### TensorBoard 监控

```bash
tensorboard --logdir runs/ldm_uncond/tb
```

记录项：`train/loss`、`train/ms_per_step`、`samples/grid`、`eval/fid`。

### FID 评测

对已有 checkpoint 评测（EMA + DDIM，参考集为当前训练图像、无翻转）：

```bash
python train.py --eval_fid --ckpt runs/ldm_uncond/ckpt.pt \
  --vq_ckpt models/vq-f4/model.ckpt --n_fid 10000
```

训练期间定期评测（默认关闭，`--fid_every 0`）：

```bash
python train.py --vq_ckpt models/vq-f4/model.ckpt --fid_every 20000 --n_fid 1000
python train.py --vq_ckpt models/vq-f4/model.ckpt --fid_final --n_fid 10000
```

## 常用参数


| 参数                 | 默认值    | 说明                                        |
| ------------------ | ------ | ----------------------------------------- |
| `--max_steps`      | 410000 | 训练步数（论文 410k；单卡建议 5–8 万）                  |
| `--batch_size`     | 48     | 论文 batch；4090 建议 8，OOM 再降 4               |
| `--lr`             | 9.6e-5 | AdamW；官方 `2e-6 × batch`，batch 8 时用 1.6e-5 |
| `--dim`            | 224    | U-Net 基础通道（约 228M；论文同 224 但结构更重）          |
| `--T`              | 1000   | 扩散步数                                      |
| `--ddim_steps`     | 200    | DDIM 子序列长度（训练偷看可用 50）                     |
| `--eta`            | 0.0    | DDIM 噪声系数：0 = 确定性，1 = DDPM 方差             |
| `--sample_every`   | 5000   | 每 N 步保存采样图（单卡建议 2000）                     |
| `--progress_every` | 10     | `--sample` 时 progression 的 DDIM 间隔        |
| `--data_dir`       | `data` | ImageFolder 根目录，或 CelebA-HQ 缓存目录          |
| `--vq_ckpt`        | 无      | 官方 vq-f4 `model.ckpt`（训练/采样/FID 必需）       |
| `--n_fid`          | 10000  | FID 生成张数                          |
| `--fid_batch`      | 8      | FID 生成 / Inception batch                        |
| `--fid_every`      | 0      | 训练期每 N 步算 FID（`0` 关闭）                       |


## 数据集


| 项目    | 本仓库                                                                                    | 论文 / 官方配置                                         |
| ----- | -------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 数据集   | [korexyz/celeba-hq-256x256](https://huggingface.co/datasets/korexyz/celeba-hq-256x256) | taming `CelebAHQTrain`（原版 CelebA-HQ resize 到 256） |
| 任务    | 无条件生成                                                                                  | 无条件生成                                             |
| 训练数据  | train + validation，约 3 万张                                                              | 仅 train split                                     |
| 分辨率   | 256×256 RGB                                                                            | 256×256 RGB                                       |
| 预处理   | Resize + CenterCrop，缩放到 **[-1, 1]**                                                    | resize 到 256，[-1, 1]                              |
| 数据增强  | 随机水平翻转                                                                                 | 随机水平翻转                                            |
| 自定义数据 | ImageFolder（`class/img`）                                                               | taming 数据模块                                       |
| FID 参考集 | 当前训练图像（磁盘原图，无翻转）                                                           | CelebA-HQ 训练集                                      |


分布接近，**不是同一划分 / 同一套原图处理**。也可把 `--data_dir` 指到自己的 ImageFolder，跳过下载。

## 实验指标

FID：生成 `--n_fid` 张样本，与当前训练图像对比（Inception-v3、2048 维）。采样使用 EMA + DDIM。论文口径为 5k 张、200 步、η=0。


| 指标                        | 本仓库 | 论文 LDM-4   |
| ------------------------- | --- | ---------- |
| **FID ↓**（DDIM，200 步，η=0） |     | **5.11**   |
| 采样墙钟                      |     | —          |
| 训练硬件 / 步数                 |     | 410k steps |


## 与论文的差异

核心流程与论文 CelebA-HQ LDM-4 一致：冻结 vq-f4、在 64×64×3 上 ε-预测 + L_simple、sqrt-space 线性 β、EMA 采样、仅 DDIM。U-Net 为便于单卡训练的轻量化实现，**不能加载官方 diffusion 权重**（vq-f4 可以）。


| 项目                  | 本仓库                                 | 论文 / 官方 LDM-4                            |
| ------------------- | ----------------------------------- | ---------------------------------------- |
| First-stage         | 官方 vq-f4（8192×3，f=4），冻结             | 同                                        |
| `encode` / `decode` | 量化前 z；decode 时 nearest lookup       | `VQModelInterface` 同                     |
| latent 尺度           | `scale_factor=1`                    | 同（CelebA 未开 `scale_by_std`）              |
| U-Net 参数量           | ~228M（`dim=224`）                    | ~274M（`openaimodel.UNetModel`，`dim=224`） |
| 每级 ResBlock / skip  | 每层 2 个 ResBlock，上采样每层 concat **一次** | 每级 2 个；上采样每个 ResBlock 都 concat           |
| 自注意力                | 每层 **一次**（32 / 16 / 8）              | 每个 ResBlock 后都有（downsample 因子 2 / 4 / 8） |
| 训练步数                | 默认 410k；单卡建议 5–8 万                  | 410k                                     |
| Batch size          | 默认 48；单卡建议 8                        | 48                                       |
| 优化器 / 学习率           | AdamW，9.6×10⁻⁵（随 batch 按 2e-6 比例改）  | AdamW，`2e-6 × batch × ngpu`              |
| EMA                 | 常数衰减 0.9999                         | 同 0.9999，另有 LitEma warmup                |
| Dropout             | 0.0                                 | 0.0                                      |
| 扩散步数 T              | 1000                                | 1000                                     |
| β schedule          | `linspace(√0.0015, √0.0195)²`       | 同                                        |
| 采样                  | 仅 DDIM                              | 论文报告 DDIM                                |
| DDIM 子序列            | 200 步、分数均匀、η=0                      | 200 步、`range(0,T,T//n)`、η=0              |
| FID / 验证集 loss      | FID 已接（规模由 `--n_fid` 控制）；无 val loss | 有                                        |
| 条件 / 其它设定           | 无                                   | 论文另含 class、text、KL、ImageNet 等            |


详细论文解读见 `docs/LDM-note.html`。

## 引用

```bibtex
@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={CVPR},
  year={2022}
}
```

