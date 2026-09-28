import torch
import os

def save_checkpoint(model, epoch, path, optimizer_G=None, optimizer_D=None, metrics=None):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'metrics': metrics
    }
    if optimizer_G: state['optim_G_state'] = optimizer_G.state_dict()
    if optimizer_D: state['optim_D_state'] = optimizer_D.state_dict()
    torch.save(state, path)

def load_checkpoint(path, model, optimizer_G=None, optimizer_D=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    # Map onto the model's device so GPU-saved checkpoints also load on CPU
    checkpoint = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer_G and 'optim_G_state' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optim_G_state'])
    if optimizer_D and 'optim_D_state' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optim_D_state'])
    return checkpoint.get('epoch', 0)

def load_model_only(path, model, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    return model.to(device).eval()

def get_checkpoint_info(path):
    checkpoint = torch.load(path, map_location='cpu')
    info = {
        'Epoch': checkpoint.get('epoch', 'Unknown'),
        'Metrics': checkpoint.get('metrics', 'N/A'),
        'Keys': list(checkpoint.keys())
    }
    return info
