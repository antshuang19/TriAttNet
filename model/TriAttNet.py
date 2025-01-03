import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(TripleAttention, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.self_att = nn.MultiheadAttention(in_channels, num_heads=4)

    def forward(self, x):
        channel_attention = self.channel_att(x).expand_as(x)
        x = x * channel_attention
        spatial_attention = self.spatial_att(x)
        x = x * spatial_attention
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, height * width).permute(0, 2, 1)  # (B, H*W, C)
        attn_output, _ = self.self_att(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)  # (B, C, H, W)
        
        return attn_output

class Residual_Block(nn.Module):
    def __init__(self, in_channel, dropout_prob=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel, eps=0.8),
            nn.PReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel, eps=0.8),
            nn.Dropout2d(dropout_prob)  
        )

    def forward(self, x):
        return x + self.block(x)
    
class TriAttNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_block=16, attention_reduction=16, dropout_prob=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        res_blocks = [Residual_Block(64, dropout_prob) for _ in range(n_residual_block)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=0.8),
        )
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=9, stride=1, padding=4)
        self.triple_attention = TripleAttention(256, reduction=attention_reduction)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        # Initial Convolution Layer
        output1 = self.conv1(x)

        # Apply Residual Blocks
        output2 = self.res_blocks(output1)

        # Second Layer Convolution
        output3 = self.conv2_1(output2) + output1
        output3 = self.conv2_2(output3)

        # Apply Triple Attention Block
        output4 = self.triple_attention(output3)

        # Upsample to original size (512x512)
        output5 = self.upsample(output4)

        # Final Output Layer
        output = self.conv4(output5)

        return output