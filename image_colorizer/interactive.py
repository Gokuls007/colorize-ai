import torch
import numpy as np
from PIL import Image
from .inference import _preprocess_image, _postprocess_output

def colorize_with_hints(image_path, hints, model, device='cpu'):
    # hints: list of (x, y, color)
    L, img_lab = _preprocess_image(image_path)
    L = L.to(device)
    
    model.eval()
    with torch.no_grad():
        ab_pred = model.net_G(L)
    
    # Simple hint blending: replace nearest pixels in ab_pred
    # (Simplified version of the real hint model)
    ab_pred_np = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    for x, y, (a, b) in hints:
        # Normalize a, b to [-1, 1]
        a_norm = a / 110.0
        b_norm = b / 110.0
        # Draw a small circle of color
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= x+i < 256 and 0 <= y+j < 256:
                    ab_pred_np[y+j, x+i] = [a_norm, b_norm]
    
    ab_pred = torch.from_numpy(ab_pred_np).permute(2, 0, 1).unsqueeze(0).to(device)
    rgb = _postprocess_output(L, ab_pred)
    return Image.fromarray((rgb * 255).astype(np.uint8))

def create_hint_template(image_path):
    img = Image.open(image_path).convert('L')
    return img
