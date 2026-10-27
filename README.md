<div align="center">

# 🧠 MAMFI

### Learning Motion at All Scales via Memory-as-Motion for Frame Interpolation

[![Checkpoints](https://img.shields.io/badge/Checkpoints-Google%20Drive-34a853.svg)](https://drive.google.com/drive/folders/1XkoBc3HrantG26B9XQweV9CwENr6RD_X?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#-license--acknowledgement)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

[Results](#-results) · [Installation](#%EF%B8%8F-installation) · [Checkpoints](#-checkpoints) · [Datasets](#-datasets) · [Benchmark](#-benchmark) · [Demo](#-demo)

</div>

---

## 📊 Results

Quantitative comparison with **Test-Time Augmentation (TTA) disabled**. Best results in **bold**.

### Standard benchmarks

| Model | Vimeo90K<br><sub>PSNR / SSIM</sub> | UCF101<br><sub>PSNR / SSIM</sub> | Xiph-2K<br><sub>PSNR / SSIM</sub> | Xiph-4K<br><sub>PSNR / SSIM</sub> | Middlebury<br><sub>IE ↓</sub> | Params<br><sub>(M)</sub> | FLOPs<br><sub>(T)</sub> |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Ours-S | 36.14 / **0.9800** | 35.43 / 0.9700 | 36.82 / 0.944 | 34.69 / 0.907 | 2.00 | **5.7** | **0.28** |
| Ours-L | **36.58** / 0.9799 | **35.51 / 0.9702** | **37.31 / 0.947** | **35.13 / 0.911** | **1.89** | 22.6 | 1.1 |

### SNU-FILM

| Model | Easy<br><sub>PSNR / SSIM</sub> | Medium<br><sub>PSNR / SSIM</sub> | Hard<br><sub>PSNR / SSIM</sub> | Extreme<br><sub>PSNR / SSIM</sub> |
|:--|:-:|:-:|:-:|:-:|
| Ours-S | 40.05 / 0.9906 | 36.18 / 0.9807 | 30.83 / 0.9399 | 25.62 / 0.8686 |
| Ours-L | **40.07 / 0.9907** | **36.30 / 0.9809** | **31.02 / 0.9417** | **25.79 / 0.8717** |

---

## 🛠️ Installation

```bash
# 1. Conda environment
conda create -n MAMFI python=3.10 -y
conda activate MAMFI

# 2. PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Dependencies
pip install -r requirements.txt
```

---

## 📦 Checkpoints

Download the pretrained weights from **[Google Drive](https://drive.google.com/drive/folders/1XkoBc3HrantG26B9XQweV9CwENr6RD_X?usp=sharing)** and place them under `./ckpt/`.

| Model | Params (M) | FLOPs (T) |
|:--|:-:|:-:|
| Ours-S | 5.7 | 0.28 |
| Ours-L | 22.6 | 1.1 |

---

## 📁 Datasets

Organize the datasets as follows (default root: `/data/datasets`, change with `--datasets_path`):

```
/data/datasets/
├── middlebury
├── snufilm
├── ucf101
├── vimeo_triplet
└── Xiph
```

| Dataset | Resolution | Test set |
|:--|:-:|:--|
| [Vimeo90K](http://toflow.csail.mit.edu/) | 448×256 | 3,782 triplets |
| [UCF101](https://liuziwei7.github.io/projects/VoxelFlow) | 256×256 | 379 triplets (from DVF) |
| [Xiph](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py) | 2K / 4K | Downsampled to 2K (Xiph-2K), center-cropped (Xiph-4K) |
| [Middlebury OTHER](https://vision.middlebury.edu/flow/data/) | ~640×480 | OTHER set |
| [SNU-FILM](https://myungsub.github.io/CAIN/) | ~1280×720 | 1,240 triplets · Easy / Medium / Hard / Extreme |

---

## 🚀 Benchmark

```bash
make bench_Ours-S
make bench_Ours-L
```

Or run directly with a custom dataset path:

```bash
python model_benchmark.py --model Ours-L --resume Ours-L --datasets_path /path/to/datasets
```

Logs are written to `./benchlog/`.

---

## 🎬 Demo

```bash
make demo-S
make demo-L
```

Input frames can be set with `--i0`, `--i1`, and `--i2` in `demo.py`.

---

## 🙏 License & Acknowledgement

This project is released under the **Apache 2.0** license.
It builds on concepts and code from [RIFE](https://github.com/hzwer/ECCV2022-RIFE), [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI), and [VFIMamba](https://github.com/MCG-NJU/VFIMamba) — please also follow their respective licenses.
We sincerely thank the authors for their excellent work, and [lucidrains](https://github.com/lucidrains) for open-source implementations that were extremely helpful.
