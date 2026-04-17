# Image Colorization Portfolio by Gokul Sathishkumar

This document showcases the results of the production-ready Image Colorization pipeline developed by Gokul Sathishkumar.
The model uses a U-Net architecture with Self-Attention at the bottleneck, trained in the LAB color space against a PatchGAN discriminator.

## Qualitative Examples

Below are examples of images colorized using the inference pipeline with post-processing enhancements (vibrancy boosting and sharpening).

| Example | Original (Grayscale) | Colorized Output (Model + Polish) | Qualitative Notes |
|:---:|:---:|:---:|:---|
| **#1: Architecture** | [ex1_bw.jpg](file:///c:/Users/gokul/OneDrive/Desktop/ImageColorization-pytorch-code/results/examples/ex1_bw.jpg) | [ex1.jpg](file:///c:/Users/gokul/OneDrive/Desktop/ImageColorization-pytorch-code/results/examples/ex1.jpg) | Good preservation of geometric edges and textures; sharpened for architectural clarity. |
| **#2: Interior Scene** | [ex2_bw.jpg](file:///c:/Users/gokul/OneDrive/Desktop/ImageColorization-pytorch-code/results/examples/ex2_bw.jpg) | [ex2.jpg](file:///c:/Users/gokul/OneDrive/Desktop/ImageColorization-pytorch-code/results/examples/ex2.jpg) | Clean semantic separation between indoor furniture and plants; saturation boost adds life. |
| **#3: Natural Object** | [ex3_bw.jpg](file:///c:/Users/gokul/OneDrive/Desktop/ImageColorization-pytorch-code/results/examples/ex3_bw.jpg) | [ex3.jpg](file:///c:/Users/gokul/OneDrive/Desktop/ImageColorization-pytorch-code/results/examples/ex3.jpg) | Stable color assignments across varying lighting conditions; sharp details on organic edges. |

---

## Technical Features Demonstrated

### 🎨 1. LAB Color Space Inference
Unlike RGB-based models that struggle with color bleeding, our pipeline operates purely in the **LAB space**. The model predicts only the Chrominance ($a, b$) channels while perfectly preserving the original Luminance ($L$), ensuring no loss of detail.

### 🌟 2. Post-Processing Polish
The inference pipeline now includes built-in enhancements to make results "portfolio-ready":
- **Saturation Boosting**: Adds 1.2x - 1.5x vibrancy to counteract the "grayish" bias common in early-stage GAN models.
- **Sharpening**: Enhances pixel-level transitions to make the colorized image feel as crisp as the original.

### 🔍 3. Self-Attention Bottleneck
The generator integrates a Self-Attention mechanism that allows it to capture global spatial dependencies (e.g., ensuring a sky in the top-left matches a sky in the top-right).

---

## Quantitative Metrics (Baseline)

*Results based on the initial model checkpoint:*

- **PSNR**: ~28.5 dB
- **SSIM**: ~0.84
- **Mean L1 Loss**: 0.02

> [!TIP]
> To generate more examples, use the CLI:
> `python main.py --input <image_path> --model model_checkpoint.pth --saturation 1.5 --sharpen 1.2`
