"""
Inference module for the ImageColorization project.

Provides functions to colorize grayscale images using a trained U-Net + PatchGAN
model. Supports both single-image and batch processing modes.
"""

import glob
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageEnhance
from skimage.color import lab2rgb, rgb2lab
from torchvision import transforms
from tqdm import tqdm

from image_colorizer.model import MainModel


def _load_model(model_path: str, device: str = "cuda") -> MainModel:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = MainModel()
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def _preprocess_image(image_path: str, size: int = 256) -> Tuple[torch.Tensor, np.ndarray]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((size, size), Image.BICUBIC)])
    img = transform(img)
    img_np = np.array(img)
    img_lab = rgb2lab(img_np).astype("float32")
    img_lab_tensor = transforms.ToTensor()(img_lab)
    L = img_lab_tensor[[0], ...] / 50.0 - 1.0
    return L.unsqueeze(0), img_lab


def _postprocess_output(L: torch.Tensor, ab_pred: torch.Tensor) -> np.ndarray:
    L_denorm = (L + 1.0) * 50.0
    ab_denorm = ab_pred * 110.0
    lab = torch.cat([L_denorm, ab_denorm], dim=1)
    lab_np = lab.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    rgb = lab2rgb(lab_np)
    return rgb


def colorize_image(
    image_path: str,
    model_path: str,
    device: str = "cuda",
    output_path: Optional[str] = None,
    size: int = 256,
    saturation_factor: float = 1.0,
    sharpen_factor: float = 1.0,
    model: Optional[MainModel] = None,
) -> Image.Image:
    if model is None:
        model = _load_model(model_path, device)

    target_device = next(model.parameters()).device
    L_tensor, _ = _preprocess_image(image_path, size)
    L_tensor = L_tensor.to(target_device)

    with torch.no_grad():
        ab_pred = model.net_G(L_tensor)

    rgb_array = _postprocess_output(L_tensor, ab_pred)
    rgb_uint8 = (np.clip(rgb_array, 0, 1) * 255).astype(np.uint8)
    result_image = Image.fromarray(rgb_uint8)

    if saturation_factor != 1.0:
        enhancer = ImageEnhance.Color(result_image)
        result_image = enhancer.enhance(saturation_factor)

    if sharpen_factor != 1.0:
        enhancer = ImageEnhance.Sharpness(result_image)
        result_image = enhancer.enhance(sharpen_factor)

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result_image.save(output_path)
        print(f"Saved colorized image to: {output_path}")

    return result_image


def colorize_batch(
    input_dir: str,
    model_path: str,
    output_dir: str,
    device: str = "cuda",
    size: int = 256,
    saturation_factor: float = 1.0,
    sharpen_factor: float = 1.0,
) -> dict:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"No images found in: {input_dir}")
        return {"total": 0, "successful": 0, "failed": 0, "errors": [], "total_time": 0.0, "avg_time": 0.0}

    model = _load_model(model_path, device)
    successful = 0
    errors = []
    start_time = time.time()

    for img_path in tqdm(image_paths, desc="Colorizing", unit="img"):
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        try:
            colorize_image(
                image_path=img_path,
                model_path=model_path,
                device=device,
                output_path=output_path,
                size=size,
                saturation_factor=saturation_factor,
                sharpen_factor=sharpen_factor,
                model=model,
            )
            successful += 1
        except Exception as e:
            errors.append((filename, str(e)))
            print(f"\nFailed to colorize '{filename}': {e}")

    total_time = time.time() - start_time
    total = len(image_paths)
    avg_time = total_time / total if total > 0 else 0.0

    print(f"\n{'='*50}\nBatch Colorization Summary\n{'='*50}")
    print(f"  Total images:   {total}")
    print(f"  Successful:     {successful}")
    print(f"  Failed:         {len(errors)}")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Avg time/image: {avg_time:.2f}s")
    print(f"{'='*50}")

    return {"total": total, "successful": successful, "failed": len(errors), "errors": errors, "total_time": total_time, "avg_time": avg_time}
