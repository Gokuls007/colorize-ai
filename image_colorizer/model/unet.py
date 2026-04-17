import torch
from torch import nn
from .attention import SelfAttention

class UnetBlock(nn.Module):
    """
    Standard recursive U-Net block (Pix2Pix style).
    Handles skip-connections by concatenating the input features with
    the features upsampled from the inner layers.
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            # Output of submodule has inner_nc * 2 channels due to skip-connections inside it
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = [down, submodule, up]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = [down, up]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]

            model = [down, submodule, up]

        # Flatten the model list for Sequential
        flattened_model = []
        for m in model:
            if isinstance(m, list):
                flattened_model.extend(m)
            else:
                flattened_model.append(m)
        self.model = nn.Sequential(*flattened_model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # Add skip connections by concatenating original input with processed output
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    """
    U-Net generator with optional Self-Attention at the bottleneck.
    """
    def __init__(self, input_c=1, output_c=2, n_filters=64, use_attention=False):
        super().__init__()
        
        # Build U-Net from the inside out
        # innermost layer
        unet_block = UnetBlock(n_filters * 8, n_filters * 8, innermost=True)
        
        # Add attention at the bottleneck if requested
        if use_attention:
            # We wrap the innermost block with a self-attention layer
            # Since innermost returns concat([x, model(x)]), it has n_filters*16 channels
            self.bottleneck_attn = SelfAttention(n_filters * 16)
            
            # Use a wrapper to apply attention to the skip-connected features
            orig_inner_forward = unet_block.forward
            def attn_forward(x):
                res = orig_inner_forward(x)
                return self.bottleneck_attn(res)
            unet_block.forward = attn_forward

        # Add middle layers
        for _ in range(3):
            unet_block = UnetBlock(n_filters * 8, n_filters * 8, submodule=unet_block, dropout=True)
        
        # Upsampling layers
        unet_block = UnetBlock(n_filters * 4, n_filters * 8, submodule=unet_block)
        unet_block = UnetBlock(n_filters * 2, n_filters * 4, submodule=unet_block)
        unet_block = UnetBlock(n_filters, n_filters * 2, submodule=unet_block)
        
        # outermost layer
        self.model = UnetBlock(output_c, n_filters, input_nc=input_c, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)
