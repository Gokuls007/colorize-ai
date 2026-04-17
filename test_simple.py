#!/usr/bin/env python
"""
Super simple test - run this and tell me which step fails
"""

print("\n" + "="*60)
print("STEP 1: Import torch")
print("="*60)
try:
    import torch
    print(" PyTorch imported successfully")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f" FAILED: {e}")
    exit(1)

print("\n" + "="*60)
print("STEP 2: Import PIL")
print("="*60)
try:
    from PIL import Image
    print(" PIL imported successfully")
except Exception as e:
    print(f" FAILED: {e}")
    exit(1)

print("\n" + "="*60)
print("STEP 3: Import numpy")
print("="*60)
try:
    import numpy as np
    print(" NumPy imported successfully")
except Exception as e:
    print(f" FAILED: {e}")
    exit(1)

print("\n" + "="*60)
print("STEP 4: Import skimage")
print("="*60)
try:
    from skimage.color import rgb2lab, lab2rgb
    print(" scikit-image imported successfully")
except Exception as e:
    print(f" FAILED: {e}")
    exit(1)

print("\n" + "="*60)
print("STEP 5: Import image_colorizer modules")
print("="*60)
try:
    from image_colorizer.model.unet import Unet
    print(" U-Net imported successfully")
except Exception as e:
    print(f" FAILED importing U-Net: {e}")
    print(f"   Make sure you're in the ImageColorization-pytorch-code directory")
    exit(1)

print("\n" + "="*60)
print("STEP 6: Import patch discriminator")
print("="*60)
try:
    from image_colorizer.model.patch_discriminator import PatchDiscriminator
    print(" PatchDiscriminator imported successfully")
except Exception as e:
    print(f" FAILED: {e}")
    exit(1)

print("\n" + "="*60)
print("STEP 7: Import MainModel")
print("="*60)
try:
    from image_colorizer.model import MainModel
    print(" MainModel imported successfully")
except Exception as e:
    print(f" FAILED: {e}")
    exit(1)

print("\n" + "="*60)
print("STEP 8: Create model instance")
print("="*60)
try:
    model = MainModel()
    print(" MainModel instance created successfully")
    print(f"   Device: {model.device}")
except Exception as e:
    print(f" FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("STEP 9: Test forward pass (dummy data)")
print("="*60)
try:
    model.net_G.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 256, 256)
        output = model.net_G(dummy_input)
    print(" Forward pass works!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output value range: [{output.min():.3f}, {output.max():.3f}]")
except Exception as e:
    print(f" FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("STEP 10: Create a simple test image")
print("="*60)
try:
    # Create a simple colored image
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_img)
    test_pil.save("test_input.jpg")
    print(" Test image created: test_input.jpg")
except Exception as e:
    print(f" FAILED: {e}")
    exit(1)

print("\n" + "="*60)
print("STEP 11: Load and process image")
print("="*60)
try:
    # Load image
    img = Image.open("test_input.jpg").convert("RGB")
    print(f" Image loaded: {img.size}")
    
    # Convert to LAB
    img_np = np.array(img).astype("float32")
    img_lab = rgb2lab(img_np)
    print(f" Converted to LAB: {img_lab.shape}")
    
    # Extract L channel
    L = img_lab[:, :, 0:1]  # Shape: (256, 256, 1)
    L_normalized = L / 50.0 - 1.0  # Normalize to [-1, 1]
    print(f" L channel extracted: {L_normalized.shape}, range: [{L_normalized.min():.2f}, {L_normalized.max():.2f}]")
    
except Exception as e:
    print(f" FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("STEP 12: Run inference on test image")
print("="*60)
try:
    from torchvision import transforms
    
    # Convert to tensor
    L_tensor = torch.from_numpy(np.transpose(L_normalized, (2, 0, 1))).unsqueeze(0)
    print(f" Tensor shape: {L_tensor.shape}")
    
    # Run inference
    model.net_G.eval()
    with torch.no_grad():
        ab_pred = model.net_G(L_tensor)
    
    print(f" Inference successful!")
    print(f"   Output shape: {ab_pred.shape}")
    print(f"   Output range: [{ab_pred.min():.3f}, {ab_pred.max():.3f}]")
    
except Exception as e:
    print(f" FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("STEP 13: Convert back to RGB")
print("="*60)
try:
    # Denormalize ab channels
    ab_denorm = ab_pred[0].cpu().numpy().transpose(1, 2, 0) * 110.0
    
    # Denormalize L channel
    # Fixed formula to match my restored pipeline exactly
    L_denorm = (L_normalized + 1.0) * 50.0
    
    # Combine
    lab_result = np.concatenate([L_denorm, ab_denorm], axis=2)
    
    # Convert to RGB
    rgb_result = lab2rgb(lab_result)
    rgb_result = (np.clip(rgb_result, 0, 1) * 255).astype(np.uint8)
    
    # Save
    result_img = Image.fromarray(rgb_result)
    result_img.save("test_output.jpg")
    
    print(f" Colorization complete!")
    print(f"   Saved to: test_output.jpg")
    print(f"   Size: {result_img.size}")
    
except Exception as e:
    print(f" FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print(" ALL TESTS PASSED!")
print("="*60)
print("\nYour model is working! Check test_output.jpg")
