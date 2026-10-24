import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
#  Self Attention
# ---------------------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=1):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True) # After attention, the numbers can become very large or unstable
        self.norm = nn.LayerNorm(embed_dim)  # Layer normalization for stability. Rescales the values to have mean 0 and variance 1
    
    def forward(self, x):
        B, C, H, W = x.shape
        if H > 128 or W > 128:
            # Downsample spatial dims to 128x128 before attention
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        # The image tensor is in the shape (B, C, H, W) - 4d. But MHA expects input in the shape (batch, sequence, embed_dim) - 3d.
        # Thus, the tensor needs to be reshaped before it is passed onto MHA
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, C, H, W)  →  (B, C, H*W)  →  (B, H*W, C)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)  # self.mha(query, key, value). The attention output tensor is stored whereas the attention weights matrix is ignored
        attn_out = self.norm(attn_out + x_flat)     # Residual connection (prevents the network from forgetting the original information while still benefiting from what attention learned.) & Norm (mean  ≈ 0 variance ≈ 1)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, 128, 128)    # (B, H*W, C)  →  (B, C, H*W)  →  (B, C, 128, 128)
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)    # (B, C, 128, 128)  →  (B, C, H, W)
        return attn_out
    
    # (B, C, H, W)        ← original input
    # (B, C, 128, 128)    ← after downsampling (if H,W > 128)
    # (B, C, 16384)       ← after view()
    # (B, 16384, C)       ← after permute() — ready for attention
    # (B, 16384, C)       ← after attention + norm
    # (B, C, 16384)       ← after permute() back
    # (B, C, 128, 128)    ← after view() back — image shape restored
    # (B, C, H, W)        ←  original image dimension restored


# ---------------------------
# Residual Dense Block (RDB)
# ---------------------------
class RDB(nn.Module):
    def __init__(self, in_channels):
        super(RDB, self).__init__()
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
        self.initial_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)    # (B, 1, H, W)  →  (B, 32, H, W)
        self.rrdb_blocks = nn.Sequential(*[RDB(embed_dim) for _ in range(num_rrdb)])    # (B, 32, H, W) →  (B, 32, H, W)
        self.attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)  # (B, 32, H, W) →  (B, 32, H, W)
        self.final_conv = nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)    # (B, 32, H, W)  →  (B, 1, H, W)

    def forward(self, x):
        initial_feature = self.initial_conv(x)    # (B, 1, H, W)  →  (B, 32, H, W)
        out = self.rrdb_blocks(initial_feature)    #  (B, 32, H, W)  →  23 RDBs  →  (B, 32, H, W)
        out = self.attention(out)  # (B, 32, H, W)  →  attention  →  (B, 32, H, W)
        out = self.final_conv(out + initial_feature)  # Global Residual connection to preserve information    (B, 32, H, W)  →  (B, 1, H, W)
        return out    # out + initial_feature  →  final_conv  →  output
