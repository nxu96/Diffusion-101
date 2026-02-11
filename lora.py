"""
lora.py - Low-Rank Adaptation (LoRA) implementation for nn.Linear layers.

This module provides:
  1. LoraLayer  - A drop-in wrapper around an existing nn.Linear that adds a
                  low-rank trainable bypass (A @ B) scaled by alpha/r.
  2. inject_lora - A utility function that replaces a named nn.Linear inside
                   a model with a LoraLayer, enabling parameter-efficient
                   fine-tuning while keeping the original weights frozen.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
"""

import math

import torch
from config import *
from torch import nn


class LoraLayer(nn.Module):
    """
    LoRA wrapper for an existing nn.Linear layer.

    The output is computed as:
        y = Linear(x)  +  x @ (A @ B) * (alpha / r)

    where A has shape (in_features, r), B has shape (r, out_features),
    and only A and B are trainable.  The original Linear weights remain
    frozen during fine-tuning.

    Args:
        raw_linear   (nn.Linear): The original linear layer to wrap.
        in_features  (int):       Input feature dimension.
        out_features (int):       Output feature dimension.
        r            (int):       Rank of the low-rank decomposition.
        alpha        (float):     Scaling factor for the LoRA output.
    """

    def __init__(self, raw_linear, in_features, out_features, r, alpha):
        super().__init__()
        # Store the rank and scaling factor for the LoRA bypass
        self.r = r
        self.alpha = alpha

        # Low-rank matrix A: (in_features, r) - initialized with Kaiming uniform
        self.lora_a = nn.Parameter(torch.empty((in_features, r)))
        # Low-rank matrix B: (r, out_features) - initialized to zeros so that
        # the LoRA contribution is zero at the start of training (no disruption
        # to the pre-trained model behavior).
        self.lora_b = nn.Parameter(torch.zeros((r, out_features)))

        # Apply Kaiming uniform initialization to matrix A (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        # Keep a reference to the original linear layer for the base forward pass
        self.raw_linear = raw_linear

    def forward(self, x):
        """
        Forward pass: base linear output + scaled low-rank bypass.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Compute the output of the original (frozen) linear layer
        raw_output = self.raw_linear(x)

        # Compute the LoRA bypass: x @ (A @ B) * (alpha / r)
        # The matrix product A @ B yields a (in_features, out_features) matrix,
        # which is then scaled by alpha/r before being multiplied with x.
        lora_output = x @ ((self.lora_a @ self.lora_b) * self.alpha / self.r)

        # Sum the original output and the LoRA contribution
        return raw_output + lora_output


def inject_lora(model, name, layer):
    """
    Replace a named nn.Linear layer inside a model with a LoraLayer.

    This function traverses the model's module hierarchy using the dotted
    `name` string (e.g. "enc_convs.0.crossattn.w_q") to locate the parent
    module, then swaps the target nn.Linear with a LoraLayer wrapper.

    Args:
        model (nn.Module): The root model containing the target layer.
        name  (str):       Dot-separated path to the layer within the model.
        layer (nn.Linear): The original linear layer to be wrapped with LoRA.
    """
    # Split the dotted name into individual component names
    name_cols = name.split(".")

    # Navigate down the module hierarchy to reach the parent of the target layer
    # e.g. for "enc_convs.0.crossattn.w_q", traverse enc_convs -> 0 -> crossattn
    children = name_cols[:-1]
    cur_layer = model
    for child in children:
        cur_layer = getattr(cur_layer, child)

    # Create a new LoraLayer wrapping the original linear layer with the
    # configured rank and alpha from config.py
    lora_layer = LoraLayer(
        layer, layer.in_features, layer.out_features, LORA_R, LORA_ALPHA
    )

    # Replace the original linear layer with the LoRA-wrapped version
    # in the parent module
    setattr(cur_layer, name_cols[-1], lora_layer)
