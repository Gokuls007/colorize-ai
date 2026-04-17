import torch
from torch import nn

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_layers=3):
        super().__init__()
        model = [nn.Conv2d(input_c, num_filters, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        model += [nn.Conv2d(num_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
