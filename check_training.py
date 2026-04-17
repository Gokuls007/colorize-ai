"""
Check if training is actually happening - ASCII ONLY VERSION
"""

import torch
import numpy as np
import os
from pathlib import Path
from image_colorizer.model import MainModel
from image_colorizer.checkpoint import load_checkpoint, load_model_only

print("=" * 60)
print("Training Diagnostic Check")
print("=" * 60)

# 1. Check if checkpoint exists
print("\n[Step 1] Check Checkpoint File:")
checkpoint_files = list(Path(".").glob("*.pth")) + list(Path("results/examples").glob("*.pth"))
if checkpoint_files:
    for f in checkpoint_files:
        size_mb = f.stat().st_size / 1e6
        print(f"   [SUCCESS] Found: {f.name} ({size_mb:.1f}MB)")
        
        # Try to load and inspect
        try:
            ckpt = torch.load(f, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                print(f"      - Epoch: {ckpt.get('epoch', 'N/A')}")
                print(f"      - Metrics: {ckpt.get('metrics', 'N/A')}")
                print(f"      - Keys: {list(ckpt.keys())[:5]}...") 
            else:
                print(f"      - Type: {type(ckpt)}")
        except Exception as e:
            print(f"      - Error loading: {e}")
else:
    print("   [CRITICAL] No checkpoint found!")
    print("      Training hasn't saved a checkpoint yet")
    exit(1)

# 2. Load model and check weights
print("\n[Step 2] Check Model Weights (on model_checkpoint.pth):")
try:
    model = MainModel()
    if Path("model_checkpoint.pth").exists():
        load_model_only("model_checkpoint.pth", model, device='cpu')
        print(f"   [SUCCESS] model_checkpoint.pth loaded")
    
    # Get weight statistics
    gen_params = list(model.net_G.parameters())
    print(f"   [INFO] Generator parameters analyzed")
    print(f"      Total layer groups: {len(gen_params)}")
    
    # Check if weights are random or learned
    first_layer = gen_params[0]
    weight_mean = first_layer.data.mean().item()
    weight_std = first_layer.data.std().item()
    
    print(f"      First layer mean: {weight_mean:.6f}")
    print(f"      First layer std: {weight_std:.6f}")
    
    if abs(weight_mean) < 0.001 and weight_std < 0.1:
        print(f"   [STATUS] Weights look RANDOM or INITIALIZED (not heavily trained)")
    else:
        print(f"   [STATUS] Weights look LEARNED (values have evolved)")
        
except Exception as e:
    print(f"   [ERROR] Error checking weights: {e}")

# 3. Run inference test
print("\n[Step 3] Test Inference Output Variance:")
try:
    model.net_G.eval()
    with torch.no_grad():
        # Create test input
        test_input = torch.randn(1, 1, 256, 256)
        output = model.net_G(test_input)
    
    # Analyze output
    out_mean = output.mean().item()
    out_std = output.std().item()
    
    print(f"   [SUCCESS] Inference successful")
    print(f"      Mean: {out_mean:.4f}")
    print(f"      Std: {out_std:.4f}")
    
    if abs(out_mean) < 0.01 and out_std < 0.01:
        print(f"   [STATUS] Output is FLAT - Model not learning!")
    elif out_std > 0.3:
        print(f"   [STATUS] Output has good variance - Generator is expressive")
    else:
        print(f"   [STATUS] Output variance is low (common in early training)")
        
except Exception as e:
    print(f"   [ERROR] Error running inference: {e}")

print("\n" + "=" * 60)
print("Diagnostic Summary Complete")
print("=" * 60)
