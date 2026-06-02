import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Multi Head Self Attention
# ---------------------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=1):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)  # Layer normalization for stability

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
    #     attn_out, _ = self.mha(x_flat, x_flat, x_flat)  # Self-attention
    #     attn_out = self.norm(attn_out + x_flat)  # Add & Norm
        
    #     attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)  # Restore shape
    #     return attn_out
    
    def forward(self, x):
        B, C, H, W = x.shape
        if H > 128 or W > 128:
            # Downsample spatial dims to 128x128 before attention
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)  # Self-attention
        attn_out = self.norm(attn_out + x_flat)     # Add & Norm
        attn_out = attn_out.permute(0, 2, 1).view(B, C, 128, 128)

        # Upsample back to original size
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)
        return attn_out


# ---------------------------
# Residual-in-Residual Dense Block (RRDB)
# ---------------------------
class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return x + out  # Residual connection


# ---------------------------
# ESRGAN Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self, in_channels=1, num_rrdb=23, embed_dim=32, num_heads=1):
        super(Generator, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(embed_dim) for _ in range(num_rrdb)])
        self.attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)  # Self-Attention Layer
        self.final_conv = nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        initial_feature = self.initial_conv(x)
        out = self.rrdb_blocks(initial_feature)
        out = self.attention(out)  # Apply Self-Attention
        out = self.final_conv(out + initial_feature)  # Residual connection to preserve information
        return out
