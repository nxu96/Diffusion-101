"""
conv_block.py - Convolutional block with time embedding injection and cross-attention.

This module defines the fundamental building block used in the UNet encoder
and decoder.  Each ConvBlock applies two convolutional layers (with group
normalization and ReLU), injects the diffusion timestep embedding between
them, and finishes with a cross-attention layer that fuses class-conditioning
information into the spatial features.
"""

from torch import nn
from cross_attn import CrossAttention


class ConvBlock(nn.Module):
    """
    A single UNet convolutional block consisting of:
      1. Conv2d -> GroupNorm -> ReLU  (changes channel count)
      2. Time embedding injection (additive, broadcast over spatial dims)
      3. Conv2d -> GroupNorm -> ReLU  (preserves channel count)
      4. Cross-attention with the class embedding

    Args:
        in_channel    (int): Number of input channels.
        out_channel   (int): Number of output channels.
        time_emb_size (int): Dimensionality of the time embedding vector.
        qsize         (int): Query/key projection size for cross-attention.
        vsize         (int): Value projection size for cross-attention.
        fsize         (int): Hidden size of the feed-forward net in cross-attention.
        cls_emb_size  (int): Dimensionality of the class embedding vector.
    """

    def __init__(self, in_channel, out_channel, time_emb_size, qsize, vsize, fsize, cls_emb_size):
        super().__init__()

        # --- First convolutional sub-block ---
        # Transforms the channel dimension from in_channel to out_channel
        # while preserving spatial dimensions (padding=1 with kernel_size=3).
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),  # Change channel count, keep spatial size
            nn.GroupNorm(32, out_channel),  # Normalize activations over groups of channels (32 groups)
            nn.ReLU(),                      # Non-linear activation
        )

        # --- Time embedding projection ---
        # Projects the time embedding vector from time_emb_size to out_channel
        # so it can be added element-wise to the feature map (broadcast across
        # the spatial width and height dimensions).
        self.time_emb_linear = nn.Linear(time_emb_size, out_channel)
        self.relu = nn.ReLU()  # Activation applied after time embedding projection

        # --- Second convolutional sub-block ---
        # Keeps both the channel count and spatial dimensions unchanged.
        self.seq2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),  # Preserve channel count and spatial size
            nn.GroupNorm(32, out_channel),  # Normalize activations over groups of channels (32 groups)
            nn.ReLU(),                      # Non-linear activation
        )

        # --- Cross-attention layer ---
        # Uses image pixels as queries and the class embedding as key/value,
        # allowing class-conditioning information to be fused into the feature map.
        # Does not change the spatial dimensions or channel count.
        self.crossattn = CrossAttention(
            channel=out_channel,
            qsize=qsize,
            vsize=vsize,
            fsize=fsize,
            cls_emb_size=cls_emb_size
        )

    def forward(self, x, t_emb, cls_emb):
        """
        Forward pass through the convolutional block.

        Args:
            x:       Input feature map of shape (batch_size, in_channel, H, W).
            t_emb:   Time embedding vector of shape (batch_size, time_emb_size).
            cls_emb: Class embedding vector of shape (batch_size, cls_emb_size).

        Returns:
            Output feature map of shape (batch_size, out_channel, H, W) with
            time and class information incorporated.
        """
        # Apply the first conv sub-block: change channel count, keep spatial dims
        x = self.seq1(x)

        # Project time embedding to out_channel and reshape for broadcasting:
        # (batch_size, time_emb_size) -> (batch_size, out_channel) -> (batch_size, out_channel, 1, 1)
        # NOTE: The time embedding is broadcast over the spatial dimensions.
        # NOTE: The time embedding is added after the first conv layer.
        # NOTE: Also need ReLU activation after the time embedding projection.
        t_emb = self.relu(self.time_emb_linear(t_emb)).view(x.size(0), x.size(1), 1, 1)

        # Add the time embedding to the feature map (broadcast over H, W) and
        # apply the second conv sub-block
        output = self.seq2(x + t_emb)

        # Fuse class-conditioning information via cross-attention and return
        return self.crossattn(output, cls_emb)
