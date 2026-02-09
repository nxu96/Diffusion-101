"""
cross_attn.py - Class-conditioned cross-attention module.

Implements a single-head cross-attention layer where image feature pixels
serve as queries and the class embedding (digit label 0-9) serves as both
key and value.  This allows the diffusion model to condition its denoising
on the desired output class.  The module follows the standard Transformer
attention pattern: Attention -> Residual + LayerNorm -> FeedForward ->
Residual + LayerNorm.
"""

import torch
from torch import nn
from config import *
import math


class CrossAttention(nn.Module):
    """
    Cross-attention block that fuses class conditioning into spatial feature maps.

    The image feature map is reshaped so that each spatial pixel becomes a
    query vector.  The class embedding is projected into a single key and a
    single value.  Because there is only one key/value pair, every pixel's
    attention score is trivially 1.0 after softmax, but the mechanism still
    provides a learnable linear mixing of class information into each pixel.

    Args:
        channel     (int): Number of channels in the input feature map.
        qsize       (int): Dimensionality of the query and key projections.
        vsize       (int): Dimensionality of the value projection.
        fsize       (int): Hidden size of the feed-forward network.
        cls_emb_size(int): Dimensionality of the class embedding vector.
    """

    def __init__(self, channel, qsize, vsize, fsize, cls_emb_size):
        super().__init__()
        # Linear projection for queries (from image pixels)
        self.w_q = nn.Linear(channel, qsize)
        # Linear projection for keys (from class embedding)
        self.w_k = nn.Linear(cls_emb_size, qsize)
        # Linear projection for values (from class embedding)
        self.w_v = nn.Linear(cls_emb_size, vsize)
        # Softmax applied to attention scores along the key dimension
        self.softmax = nn.Softmax(dim=-1)
        # Linear layer to project attention output back to the channel dimension
        self.z_linear = nn.Linear(vsize, channel)
        # Layer normalization after attention residual connection
        self.norm1 = nn.LayerNorm(channel)
        # Position-wise feed-forward network (two linear layers with ReLU)
        self.feedforward = nn.Sequential(
            nn.Linear(channel, fsize),  # Expand to feed-forward hidden size
            nn.ReLU(),                  # Non-linear activation
            nn.Linear(fsize, channel)   # Project back to channel dimension
        )
        # Layer normalization after feed-forward residual connection
        self.norm2 = nn.LayerNorm(channel)

    def forward(self, x, cls_emb):
        """
        Forward pass for the cross-attention block.

        Args:
            x:       Feature map tensor of shape (batch_size, channel, width, height).
            cls_emb: Class embedding tensor of shape (batch_size, cls_emb_size).

        Returns:
            Output tensor of shape (batch_size, channel, width, height) with class
            information fused into the spatial features.
        """
        # Rearrange from (B, C, W, H) to (B, W, H, C) so the channel dim is last,
        # which is required for the linear projections.
        x = x.permute(0, 2, 3, 1)  # x: (batch_size, width, height, channel)

        # --- Query projection (from image pixels) ---
        Q = self.w_q(x)  # Q: (batch_size, width, height, qsize)
        # Flatten spatial dimensions into a single sequence dimension
        Q = Q.view(Q.size(0), Q.size(1) * Q.size(2), Q.size(3))  # Q: (batch_size, width*height, qsize)

        # --- Key projection (from class embedding) ---
        K = self.w_k(cls_emb)  # K: (batch_size, qsize)
        # Reshape for batched matrix multiplication: (batch_size, qsize) -> (batch_size, qsize, 1)
        K = K.view(K.size(0), K.size(1), 1)  # K: (batch_size, qsize, 1)

        # --- Value projection (from class embedding) ---
        V = self.w_v(cls_emb)  # V: (batch_size, vsize)
        # Reshape for batched matrix multiplication: (batch_size, vsize) -> (batch_size, 1, vsize)
        V = V.view(V.size(0), 1, V.size(1))  # V: (batch_size, 1, vsize)

        # --- Scaled dot-product attention ---
        # Compute attention scores: Q @ K / sqrt(d_k)
        attn = torch.matmul(Q, K) / math.sqrt(Q.size(2))  # attn: (batch_size, width*height, 1)
        # Apply softmax to normalize scores (trivially 1.0 since there is only one key)
        attn = self.softmax(attn)  # attn: (batch_size, width*height, 1)

        # --- Attention output ---
        # Weighted sum of values (effectively scales V by the attention score)
        Z = torch.matmul(attn, V)  # Z: (batch_size, width*height, vsize)
        # Project attention output back to channel dimension
        Z = self.z_linear(Z)  # Z: (batch_size, width*height, channel)
        # Restore spatial dimensions
        Z = Z.view(x.size(0), x.size(1), x.size(2), x.size(3))  # Z: (batch_size, width, height, channel)

        # --- First residual connection + layer normalization ---
        Z = self.norm1(Z + x)  # Z: (batch_size, width, height, channel)

        # --- Feed-forward network ---
        out = self.feedforward(Z)  # out: (batch_size, width, height, channel)

        # --- Second residual connection + layer normalization ---
        out = self.norm2(out + Z)

        # Rearrange back to (B, C, W, H) for downstream convolutional layers
        return out.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Quick sanity-check (runs only when executed as a script)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Define a small test configuration
    batch_size = 2
    channel = 1
    qsize = 256
    cls_emb_size = 32

    # Instantiate the cross-attention module
    cross_atn = CrossAttention(channel=1, qsize=256, vsize=128, fsize=512, cls_emb_size=32)

    # Create dummy input tensors
    x = torch.randn((batch_size, channel, IMG_SIZE, IMG_SIZE))     # Random feature map
    cls_emb = torch.randn((batch_size, cls_emb_size))               # Random class embeddings

    # Forward pass and print output shape (expected: (2, 1, 48, 48))
    Z = cross_atn(x, cls_emb)
    print(Z.size())  # Z: (2, 1, 48, 48)
