"""
Smoke tests for the inference pipeline.

Run with:  python -m pytest smoke_test.py
Skip the test that needs the real checkpoint:  python -m pytest smoke_test.py -m "not model"
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from image_colorizer.inference import (
    _is_lfs_pointer,
    _load_model,
    _postprocess_output,
    _preprocess_image,
    colorize_image,
)
from image_colorizer.model.unet import Unet

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model_checkpoint.pth")
SAMPLE_BW = os.path.join(ROOT, "results", "examples", "ex1_bw.jpg")
HAS_WEIGHTS = os.path.exists(MODEL_PATH) and not _is_lfs_pointer(MODEL_PATH)


@pytest.mark.parametrize("use_attention", [False, True])
def test_generator_output_shape(use_attention):
    net = Unet(input_c=1, output_c=2, use_attention=use_attention).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 1, 256, 256))
    assert out.shape == (1, 2, 256, 256)
    assert out.abs().max() <= 1.0


def test_lab_roundtrip(tmp_path):
    rgb = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
    path = tmp_path / "rand.png"
    Image.fromarray(rgb).save(path)

    L, img_lab = _preprocess_image(str(path), size=32)
    assert L.shape == (1, 1, 32, 32)
    assert L.min() >= -1.0 and L.max() <= 1.0

    ab = torch.from_numpy(img_lab[:, :, 1:]).permute(2, 0, 1).unsqueeze(0) / 110.0
    rgb_back = _postprocess_output(L, ab)
    expected = np.array(Image.fromarray(rgb).resize((32, 32), Image.BICUBIC)) / 255.0
    assert np.abs(rgb_back - expected).max() < 0.02


def test_lfs_pointer_is_reported(tmp_path):
    pointer = tmp_path / "model_checkpoint.pth"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000\nsize 228789723\n"
    )
    assert _is_lfs_pointer(str(pointer))
    with pytest.raises(RuntimeError, match="git lfs pull"):
        _load_model(str(pointer), device="cpu")


@pytest.mark.model
def test_colorize_keeps_size_and_adds_color(tmp_path):
    if not HAS_WEIGHTS:
        # CI sets REQUIRE_MODEL=1 on the job that fetches the LFS checkpoint,
        # so a failed download fails the build instead of silently skipping
        if os.environ.get("REQUIRE_MODEL") == "1":
            pytest.fail("model_checkpoint.pth is missing or an LFS pointer")
        pytest.skip("model_checkpoint.pth not downloaded (run 'git lfs pull')")
    src = Image.open(SAMPLE_BW)
    assert src.mode == "L"

    out_path = tmp_path / "out.jpg"
    result = colorize_image(SAMPLE_BW, MODEL_PATH, device="cpu", output_path=str(out_path))

    assert result.mode == "RGB"
    assert result.size == src.size
    assert out_path.exists()
    arr = np.asarray(result).astype(np.int16)
    chroma = np.abs(arr[..., 0] - arr[..., 2]).mean()
    assert chroma > 1.0, "output is still grayscale"

    # A PIL image (e.g. an RGBA upload from Streamlit) works without a temp file
    rgba = src.convert("RGBA")
    assert colorize_image(rgba, MODEL_PATH, device="cpu").size == src.size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
