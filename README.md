# 🎨 ImageColorization — Deep Learning Powered Photo Revitalization

[![CI](https://github.com/Gokuls007/colorize-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Gokuls007/colorize-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**ImageColorization** is a production-grade deep learning pipeline designed to breathe new life into black and white images. Using a state-of-the-art **Pix2Pix GAN architecture** (U-Net Generator + PatchGAN Discriminator) operating in the **CIE LAB color space**, it automatically predicts realistic chromaticity for grayscale inputs.

> Rediscover history in vibrant color. Perfect for old family photos, historical archives, and creative film projects.

---

## ⚡ Quick Start

All options need the trained weights. `model_checkpoint.pth` (~230 MB) is stored with Git LFS, so clone with Git LFS installed:

```bash
git lfs install
git clone https://github.com/Gokuls007/colorize-ai.git
cd colorize-ai
```

**Docker** (CPU only, no local Python needed):
```bash
docker build -t colorize-ai .
docker run --rm -p 8501:8501 colorize-ai
# then open http://localhost:8501
```

**Local Streamlit app:**
```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

**Command line:**
```bash
# Single image (output keeps the input's size)
python main.py --input photo.jpg --output photo_color.jpg

# Every .jpg/.jpeg/.png in a folder
python main.py --input ./old_photos --output ./restored --batch
```

`--model` defaults to `model_checkpoint.pth`. `--device` accepts `cuda`, `mps` or `cpu` and falls back to CPU if the requested device is not available.

---

## 🖼️ Results Gallery

Here is a glimpse of what the model can achieve at **Epoch 4**. For a full set of high-resolution comparisons, see the [Results Directory](results/README.md).

| Original (B&W) / Predicted Color | Scene Type |
|-----------------------------------|------------|
| ![Sample 1](results/result_1_comparison.jpg) | Natural Landscapes |
| ![Sample 2](results/result_2_comparison.jpg) | Indoor Textures |
| ![Sample 3](results/result_3_comparison.jpg) | Complex Structural Details |

---

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Intelligent Inference** | High-fidelity colorization for single images or complete directories. |
| 🌐 **Interactive Dashboard** | Professional Streamlit web app for real-time experimentation. |
| 📊 **Scientific Metrics** | Integrated PSNR and SSIM calculation for objective quality assessment. |
| 💾 **Advanced Checkpointing** | Robust save/load system for seamless training continuation. |
| 🚀 **Hardware Agnostic** | Optimized for CUDA, Apple Silicon (MPS), and ultra-optimized CPU fallback. |

---

## 🏗️ How It Works

The system utilizes a **Conditional Generative Adversarial Network (cGAN)** specialized for image-to-image translation.

### 1. The Color Space (LAB)
Unlike RGB, the **LAB color space** separates Lightness (L) from chromaticity (A and B). This is ideal for colorization because:
- The **L-channel** provides a perfect grayscale input.
- The model only needs to predict the 2-channel **ab** distribution, reducing complexity.

### 2. Architecture Details
- **Generator (U-Net)**: Uses symmetrical skip connections to pass high-frequency details from the encoder directly to the decoder, preventing blurriness.
- **Discriminator (PatchGAN)**: Instead of judging the whole image, it classifies whether each 30x30 patch is "real" or "fake", forcing the generator to produce sharp, local color borders.

---

## 🚀 Getting Started

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Gokuls007/colorize-ai.git
cd colorize-ai

# 2. Download the trained weights (model_checkpoint.pth is stored with Git LFS)
git lfs install
git lfs pull

# 3. Install core dependencies
pip install -r requirements.txt
```

> If `model_checkpoint.pth` is only a few hundred bytes, it is a Git LFS pointer rather than the real weights. Run `git lfs pull` to fetch the ~230 MB checkpoint.

### Usage

#### 🖥️ Web Interface (Recommended)
Launch the interactive dashboard to colorize images via drag-and-drop:
```bash
python -m streamlit run streamlit_app.py
```

#### ⌨️ Command Line
```bash
# Single image
python main.py --input photo.jpg --model model_checkpoint.pth --output result.jpg

# Batch process a folder
python main.py --input ./old_photos --model model_checkpoint.pth --output ./restored --batch
```

---

## 🛠️ Performance & Training

| Parameter | Value |
|-----------|-------|
| **Backbone** | Pix2Pix U-Net generator + PatchGAN discriminator |
| **Model Resolution** | 256 x 256 (colour is upsampled back to the input size) |
| **Dataset** | COCO Sample (2017) |
| **Included Checkpoint** | `model_checkpoint.pth`, epoch 6 |
| **Inference Time (CPU)** | ~0.2 s for a 640x428 image, model already loaded (24-thread desktop CPU, PyTorch 2.14 CPU build); GPU not measured |

**Accuracy metrics:** PSNR/SSIM have not been measured for the included checkpoint yet. The repository only contains grayscale inputs, with no colour ground truth to compare against. To measure them on a COCO validation split, run `python main.py --evaluate`. This downloads the COCO sample dataset.

---

## 📝 Acknowledgments

- **Richard Zhang et al.** for the pioneering work on automatic colorization (ECCV 2016).
- **Phillip Isola et al.** for the Pix2Pix architecture (CVPR 2017).
- **FastAI Team** for the public COCO dataset samples.

---
<p align="center">
  <b>Developed by Gokul Sathishkumar — Bringing the past into the present.</b>
</p>
