"""
config.py - Global configuration constants for the Diffusion-101 project.

This module defines hyperparameters and settings shared across training,
inference, and model architecture modules, including image dimensions,
diffusion schedule length, LoRA parameters, and compute device selection.
"""

import torch

# The spatial resolution (width and height) to which all input images are resized.
# MNIST images (originally 28x28) are upscaled to 48x48 for richer spatial features.
IMG_SIZE = 48

# Total number of diffusion timesteps used in the forward (noising) and
# reverse (denoising) processes.  A larger T yields finer noise granularity
# but increases training and inference time.
T = 1000

# LoRA (Low-Rank Adaptation) scaling factor.  The LoRA output is multiplied
# by (alpha / r) before being added to the original linear layer output.
# A value of 1 means the LoRA contribution is scaled purely by 1/r.
LORA_ALPHA = 1

# LoRA rank – the bottleneck dimension of the low-rank decomposition.
# Smaller r means fewer trainable parameters; larger r increases capacity.
LORA_R = 8

# Automatically select GPU (CUDA) if available; otherwise fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
