"""
time_position_emb.py - Sinusoidal positional (time-step) embedding for diffusion models.

Implements the same positional encoding scheme used in the original Transformer
paper ("Attention Is All You Need"), adapted here to encode the diffusion
timestep t into a fixed-size vector.  The embedding allows the model to
distinguish between different noise levels during training and inference.
"""

import math

import torch
from config import *
from torch import nn


class TimePositionEmbedding(nn.Module):
    """
    Sinusoidal time-step embedding.

    Given integer timesteps t in [0, T), this module produces a continuous
    vector of size `emb_size` using interleaved sine and cosine functions at
    geometrically spaced frequencies – identical to the positional encoding
    in Vaswani et al. (2017).

    Args:
        emb_size (int): Total embedding dimension (must be even).
    """

    def __init__(self, emb_size):
        super().__init__()
        # We split the embedding into two halves: one for sin, one for cos.
        self.half_emb_size = emb_size // 2

        # Precompute the geometric frequency series:
        #   freq_i = exp( i * (-ln(10000) / (d/2 - 1)) )   for i = 0 .. d/2-1
        # This yields frequencies that span several orders of magnitude,
        # enabling the model to capture both fine and coarse temporal information.
        half_emb = torch.exp(
            torch.arange(self.half_emb_size)
            * (-1 * math.log(10000) / (self.half_emb_size - 1))
        )

        # Register as a buffer so it moves with the module to GPU/CPU but is
        # not treated as a learnable parameter.
        self.register_buffer("half_emb", half_emb)

    def forward(self, t):
        """
        Compute the sinusoidal embedding for a batch of timesteps.

        Args:
            t: Integer tensor of shape (batch_size,) containing timestep indices.

        Returns:
            Tensor of shape (batch_size, emb_size) with the positional embeddings.
        """
        # Reshape t from (batch_size,) to (batch_size, 1) for broadcasting.
        t = t.view(t.size(0), 1)

        # Expand the frequency vector to match the batch dimension:
        # (1, half_emb_size) -> (batch_size, half_emb_size)
        half_emb = self.half_emb.unsqueeze(0).expand(t.size(0), self.half_emb_size)

        # Element-wise product of timestep and frequency: t * freq_i
        half_emb_t = half_emb * t

        # Concatenate sin and cos components along the last dimension to form
        # the full embedding of size emb_size.
        embs_t = torch.cat((half_emb_t.sin(), half_emb_t.cos()), dim=-1)

        return embs_t


# ---------------------------------------------------------------------------
# Quick sanity check (runs only when executed as a script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a TimePositionEmbedding module with embedding size 8, moved to DEVICE.
    time_pos_emb = TimePositionEmbedding(8).to(DEVICE)

    # Generate 2 random timestep indices in [0, T) to simulate a mini-batch.
    t = torch.randint(0, T, (2,)).to(DEVICE)
    print("timesteps are ", t)

    # Compute and print the resulting embeddings.
    embs_t = time_pos_emb(t)
    print("time embedding shape is ", embs_t.shape)  # (bs, emb_size)
    print("time embedding is ", embs_t)
