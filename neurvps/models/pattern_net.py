"""
pattern_net.py

End-to-end trainable pattern feature extractor.
Produces a dense (N, embed_dim, H, W) feature map
in the same format as the NeurVPS hourglass backbone.

Every pixel gets a feature vector that encodes:
    1. What the local patch at this pixel looks like
       (from the CNN encoder)
    2. Where similar-looking patches are elsewhere
       in the image (from the transformer)

This is the learned equivalent of SIFT + clustering,
but with a fixed, controllable output shape that is
compatible with conic convolution.
"""

import math
from typing import OrderedDict
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 4, 8, 16)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        # Fuse all branches back to out_ch
        self.fuse = ConvBnRelu(out_ch * len(rates), out_ch, kernel_size=1,
                               padding=0)

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))

# -----------------------------------------------------------------------
# CNN Encoder
# -----------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Standard conv → BN → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        print("checking ",x.shape, x.is_contiguous(), x.dtype, x.device)
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Two conv layers with a skip connection.
    Keeps spatial size and channel count the same.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBnRelu(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + x)


class PatternCNNEncoder(nn.Module):
    """
    Encodes the input image into a dense spatial feature map.

    Output spatial size matches the NeurVPS backbone output:
        Input:  (N, 3, 512, 512)
        Output: (N, embed_dim, 128, 128)

    Two stride-2 pooling operations give the 4× downsampling
    that matches NeurVPS's output_stride=4.

    Each output pixel has a receptive field covering roughly
    a 64×64 region of the input — large enough to capture
    recurring visual elements like windows, columns, tiles.
    """
    def __init__(self, embed_dim=64):
        super().__init__()

        # Stage 1: 512×512 → 256×256, 32 channels
        # Captures fine-grained local appearance
        self.stage1 = nn.Sequential(
            ConvBnRelu(3, 32, kernel_size=7, stride=2, padding=3),
            ResidualBlock(32),
            ResidualBlock(32),
        )

        # Stage 2: 256×256 → 128×128, embed_dim channels
        # Captures broader patch appearance context
        # Output spatial size matches backbone exactly
        self.stage2 = nn.Sequential(
            ConvBnRelu(32, embed_dim, stride=2),
            ResidualBlock(embed_dim),
            ResidualBlock(embed_dim),
        )
        self.aspp = ASPP(embed_dim, embed_dim)

    def forward(self, x):
        """
        x      : (N, 3, 512, 512)
        returns: (N, embed_dim, 128, 128)
        """
        x = self.stage1(x)   # (N, 32,        256, 256)
        x = self.stage2(x)   # (N, embed_dim, 128, 128)
        x = self.aspp(x)
        return x


# -----------------------------------------------------------------------
# Positional Encoding
# -----------------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding.

    Gives the transformer spatial awareness — without this,
    the transformer has no way to reason about whether
    similar patches are spatially arranged in a perspective-
    consistent way (i.e. along a line toward the VP).

    Encodes row and column positions separately using
    sin/cos at multiple frequencies, then concatenates.

    Taken from learnopencv.com
    """
    def __init__(self, embed_dim, grid_H=128, grid_W=128):
        super().__init__()
        half = embed_dim // 2

        div = torch.exp(
            torch.arange(0, half, 2).float()
            * (-math.log(10000.0) / half)
        )

        # Row encoding
        rows = torch.arange(grid_H).float().unsqueeze(1)  # (H, 1)
        row_enc = torch.zeros(grid_H, half)
        row_enc[:, 0::2] = torch.sin(rows * div)
        row_enc[:, 1::2] = torch.cos(rows * div)

        # Column encoding
        cols = torch.arange(grid_W).float().unsqueeze(1)  # (W, 1)
        col_enc = torch.zeros(grid_W, half)
        col_enc[:, 0::2] = torch.sin(cols * div)
        col_enc[:, 1::2] = torch.cos(cols * div)

        # Combine for every (row, col) position
        # row_enc[r] concatenated with col_enc[c]
        row_exp = row_enc.unsqueeze(1).expand(-1, grid_W, -1)  # (H, W, half)
        col_exp = col_enc.unsqueeze(0).expand(grid_H, -1, -1)  # (H, W, half)
        pos = torch.cat([row_exp, col_exp], dim=-1)            # (H, W, embed_dim)
        pos = pos.view(1, grid_H * grid_W, embed_dim)          # (1, H*W, embed_dim)

        # Buffer: moves with model device, not a learned parameter
        self.register_buffer("pos", pos)

    def forward(self, tokens):
        """
        tokens : (N, H*W, embed_dim)
        returns: (N, H*W, embed_dim)  with position information added
        """
        return tokens + self.pos


# -----------------------------------------------------------------------
# Transformer
# -----------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """
    Single pre-norm transformer block.
    Pre-norm (LN before attention) is more stable than post-norm.

    Self-attention here is doing the key work: pixels with similar
    CNN features attend strongly to each other. After this layer,
    each pixel's feature vector reflects not just "what I look like"
    but "where similar-looking pixels are in this image."

    taken from teh pytorch ViT transformer encoder
    """
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        #attention block
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
       # FFN block 
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        """input: (N, seq_len, embed_dim)"""
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # Self-attention with pre-norm
        x = self.norm1(input)
        x, _ = self.attn(x, x, x, need_weights = False)
        x = self.dropout(x)
        x = x + input
        # FFN with pre-norm
        y = self.norm2(x)
        y = self.ffn(y)
        return x + y


# -----------------------------------------------------------------------
# Full Pattern Network
# -----------------------------------------------------------------------

class PatternNet(nn.Module):
    """
    Full pattern feature extractor.

    Pipeline:
        Image
          → CNN encoder     (local patch appearance per pixel)
          → flatten to seq
          → pos encoding    (spatial location per pixel)
          → transformer     (recurring pattern context per pixel)
          → reshape to map  (N, embed_dim, H, W)

    Output is a dense spatial feature map in exactly the same
    format as the NeurVPS hourglass backbone output. It can be
    concatenated directly with backbone features and passed to
    the modified ApolloniusNet.

    The network is fully differentiable and trained end-to-end
    with NeurVPS using only the VP classification loss.
    No manual vocabulary, no keypoint detection, no RANSAC.
    """
    def __init__(
        self,
        embed_dim=64,
        num_heads=8,
        num_layers=2,
        ffn_dim=256,
        dropout=0.1,
        grid_H=128,
        grid_W=128,
    ):
        super().__init__()
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.embed_dim = embed_dim

        # CNN: local appearance per pixel
        self.cnn = PatternCNNEncoder(embed_dim)

        # Positional encoding: spatial location per pixel
        self.pos_enc = PositionalEncoding2D(embed_dim, grid_H, grid_W)

        # Transformer: recurring pattern context
        # Shallow (2 layers) — we only need enough depth to propagate
        # similarity information across the spatial map.
        # VP reasoning is handled by conic convolution, not here.
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = TransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
        self.layers = nn.Sequential(layers)    
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image):
        """
        image  : (N, 3, 512, 512)
        returns: (N, embed_dim, 128, 128)

        Output shape is always fixed regardless of image content.
        Every pixel has a feature vector. Compatible with conic conv.
        """
        N = image.shape[0]

        # Step 1: CNN — local appearance at every pixel
        
        x = self.cnn(image)                          # (N, embed_dim, H, W)
        _, C, H, W = x.shape
        print("checking ",x.shape, x.is_contiguous(), x.dtype, x.device)
        # Step 2: Flatten to sequence for transformer
        tokens = x.flatten(2).transpose(1, 2)        # (N, H*W, embed_dim)

        #from here adapted from pytorch vit encoder
        # Step 3: Add positional encoding
        tokens = self.pos_enc(tokens)                # (N, H*W, embed_dim)

        # # Step 4: Transformer — each pixel now also encodes
        # # where similar pixels are (recurring pattern context)
        # for layer in self.transformer:
        #     tokens = layer(tokens)                   # (N, H*W, embed_dim)
        # tokens = self.norm(tokens)

        # # Step 5: Reshape back to spatial feature map
        # # Same format as backbone output — ready for concatenation
        # out = tokens.transpose(1, 2).view(N, C, H, W)   # (N, embed_dim, H, W)
        tokens = self.norm(self.layers(self.dropout(tokens)))
        out = tokens.transpose(1, 2).view(N, C, H, W)
        return out
