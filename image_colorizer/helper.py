def update_losses(model, loss_meter_dict, count=1):
    for key in ['loss_G', 'loss_D', 'loss_G_GAN', 'loss_G_L1']:
        val = getattr(model, key).item()
        loss_meter_dict[key].update(val, count)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count
