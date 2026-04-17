import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

def visualize(model, data, save_path=None):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    
    fake_imgs = _lab_to_rgb(L, fake_color)
    real_imgs = _lab_to_rgb(L, real_color)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(real_imgs[0])
    axes[0].set_title("Real")
    axes[1].imshow(fake_imgs[0])
    axes[1].set_title("Fake")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def _lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        rgb_imgs.append(lab2rgb(img))
    return np.stack(rgb_imgs, axis=0)
