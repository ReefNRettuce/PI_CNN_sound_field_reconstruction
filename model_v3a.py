# model_v3a.py
# PICNN U-Net - SMALLER VERSION (~600k params instead of 4.6M)

import torch 
import torch.nn as nn
import torch.nn.functional as F


class decoder_double_convolution(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.convolution_operation = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
    
    def forward(self, x):
        return self.convolution_operation(x)


class decoder_final_layer_convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.final_conv(x)


class partial_convolutions(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.register_buffer('mask_weight', torch.ones(1, 1, kernel_size, kernel_size))
        self.partial_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x, mask_input):
        mask_broadcast = mask_input.repeat(1, self.in_channels, 1, 1)
        masked_x = x * mask_broadcast
        product = self.conv(masked_x)

        with torch.no_grad():
            mask_sum = F.conv2d(mask_input, self.mask_weight.to(x.device), stride=self.stride, padding=self.padding)
            mask_sum_no_zero = mask_sum.clone()
            mask_sum_no_zero[mask_sum == 0] = 1
            scaling_factor = (self.kernel_size ** 2) / mask_sum_no_zero

        out = product * scaling_factor
        mask_out = (mask_sum > 0).float()
        out = self.partial_bn(out)
        out = self.activation(out)

        return out, mask_out


class tiny_unet_small(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_1 = partial_convolutions(8, 16, kernel_size=5, stride=2, padding=2, negative_slope=0.2)
        self.encoder_2 = partial_convolutions(16, 32, kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        self.encoder_3 = partial_convolutions(32, 64, kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        self.bottle_neck = partial_convolutions(64, 128, kernel_size=3, stride=2, padding=1, negative_slope=0.2)
        
        # Decoder
        self.up_transpose_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_1 = decoder_double_convolution(128, 64, dropout_rate)
        
        self.up_transpose_2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_2 = decoder_double_convolution(64, 32, dropout_rate)
        
        self.up_transpose_3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder_3 = decoder_double_convolution(32, 16, dropout_rate)
        
        self.final_upsample = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.decoder_output = decoder_final_layer_convolution(8, out_channels)

    def forward(self, x, mask):
        x_init = self.initial_conv(x)
        
        x_1, m_1 = self.encoder_1(x_init, mask)
        x_2, m_2 = self.encoder_2(x_1, m_1)
        x_3, m_3 = self.encoder_3(x_2, m_2)
        x_bot, m_bot = self.bottle_neck(x_3, m_3)
        
        x = self.up_transpose_1(x_bot)
        x = torch.cat([x, x_3], dim=1)
        x = self.decoder_1(x)
        
        x = self.up_transpose_2(x)
        x = torch.cat([x, x_2], dim=1)
        x = self.decoder_2(x)
        
        x = self.up_transpose_3(x)
        x = torch.cat([x, x_1], dim=1)
        x = self.decoder_3(x)
        
        x = self.final_upsample(x)
        x = self.decoder_output(x)
        return x


if __name__ == '__main__':
    model = tiny_unet_small(in_channels=2, out_channels=8, dropout_rate=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    dummy_input = torch.randn(2, 2, 32, 32)
    dummy_mask = torch.ones(2, 1, 32, 32)
    output = model(dummy_input, dummy_mask)
    print(f"Output shape: {output.shape}")