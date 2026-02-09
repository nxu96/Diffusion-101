"""
diffusion.py - Forward diffusion process for DDPM (Denoising Diffusion Probabilistic Models).

This module implements the forward (noising) process of diffusion models.
It precomputes the noise schedule parameters (betas, alphas, cumulative
products, and posterior variance) and provides a function that, given a
clean image batch and a batch of timesteps, directly produces the noised
images and the corresponding noise tensors used as training targets.
"""

import matplotlib.pyplot as plt
import torch
from config import *
from dataset import tensor_to_pil, train_dataset

# ===========================================================================
# Noise schedule precomputation
# ===========================================================================

# Linear beta schedule: noise variance increases linearly from 0.0001 to 0.02
# over T timesteps.  Shape: (T,)
# Beta is the noise variance at each timestep.
betas = torch.linspace(0.0001, 0.02, T)

# Alpha values: alpha_t = 1 - beta_t.  These represent the fraction of
# signal retained at each step.  Shape: (T,)
# So when the beta increases, we mix more noise into the image.
alphas = 1 - betas

# Cumulative product of alphas: \bar{alpha}_t = \prod_{s=1}^{t} alpha_s
# This lets us jump directly from the clean image x_0 to any noisy x_t
# in a single step (the "nice property" of DDPM).
# Example: [a1, a2, a3, ...] -> [a1, a1*a2, a1*a2*a3, ...]
# Shape: (T,)
alphas_cumprod = torch.cumprod(alphas, dim=-1)

# Shifted cumulative product: \bar{alpha}_{t-1}, padded with 1.0 at the start.
# Used in the reverse process to compute the posterior mean and variance.
# Example: [1, a1, a1*a2, a1*a2*a3, ...]
# Shape: (T,)
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1)

# Posterior variance used during the reverse (denoising) process:
#   \tilde{beta}_t = beta_t * (1 - \bar{alpha}_{t-1}) / (1 - \bar{alpha}_t)
# This is the variance of the Gaussian transition q(x_{t-1} | x_t, x_0).
# Shape: (T,)
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)


# ===========================================================================
# Forward diffusion function
# ===========================================================================


def forward_diffusion(batch_x, batch_t):
    """
    Apply the forward diffusion process to produce noised images at
    arbitrary timesteps in a single step (closed-form).

    The key formula is:
        x_t = sqrt(\bar{alpha}_t) * x_0  +  sqrt(1 - \bar{alpha}_t) * epsilon

    where epsilon ~ N(0, I).

    Args:
        batch_x: Clean image batch of shape (batch, channel, height, width)
                 with pixel values in [-1, 1].
        batch_t: Integer timestep indices of shape (batch_size,), one per image.

    Returns:
        batch_x_t:     Noised images at the given timesteps.
                       Shape: (batch, channel, height, width).
        batch_noise_t: The Gaussian noise that was added (training target).
                       Shape: (batch, channel, height, width).
    """
    # Sample standard Gaussian noise with the same shape as the input images
    batch_noise_t = torch.randn_like(batch_x)  # (batch, channel, height, width)

    # Gather \bar{alpha}_t for each image in the batch and reshape for
    # broadcasting over (channel, height, width) dimensions.
    batch_alphas_cumprod = alphas_cumprod.to(DEVICE)[batch_t].view(
        batch_x.size(0), 1, 1, 1
    )

    # Directly compute x_t using the closed-form reparameterization:
    #   x_t = sqrt(\bar{alpha}_t) * x_0  +  sqrt(1 - \bar{alpha}_t) * noise
    # Element-wise multiplication and addition.
    batch_x_t = (
        torch.sqrt(batch_alphas_cumprod) * batch_x
        + torch.sqrt(1 - batch_alphas_cumprod) * batch_noise_t
    )

    return batch_x_t, batch_noise_t


# ===========================================================================
# Visualization demo (runs only when executed as a script)
# ===========================================================================
if __name__ == "__main__":
    # Build a mini-batch of 2 images from the training set
    batch_x = torch.stack((train_dataset[0][0], train_dataset[1][0]), dim=0).to(
        DEVICE
    )  # (2, 1, 48, 48)

    # --- Display original (clean) images ---
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(tensor_to_pil(batch_x[0]))
    # plt.subplot(1, 2, 2)
    # plt.imshow(tensor_to_pil(batch_x[1]))
    # plt.show()
    # plt.savefig("imgs/forward_diffusion_original_images.png")
    # Rescale pixel values from [0, 1] to [-1, 1] to match the Gaussian noise range
    batch_x = batch_x * 2 - 1

    # Sample random timesteps for each image in the batch
    batch_t = torch.randint(0, T, size=(batch_x.size(0),)).to(DEVICE)
    # batch_t = torch.tensor([10, 100], dtype=torch.long).to(DEVICE)
    print("batch_t:", batch_t)

    # Apply the forward diffusion to get noised images and the noise
    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)
    print("batch_x_t:", batch_x_t.size())
    print("batch_noise_t:", batch_noise_t.size())

    # --- Display noised images ---
    # Rescale from [-1, 1] back to [0, 1] for visualization
    plt.figure(figsize=(10, 10))
    # Original images
    plt.subplot(2, 2, 1)
    plt.title("Original Image 1")
    plt.imshow(tensor_to_pil((batch_x[0] + 1) / 2))
    plt.subplot(2, 2, 3)
    plt.title("Original Image 2")
    plt.imshow(tensor_to_pil((batch_x[1] + 1) / 2))
    # Noised images
    plt.subplot(2, 2, 2)
    plt.title(f"Noised Image 1 at t = {batch_t[0].item()}")
    plt.imshow(tensor_to_pil((batch_x_t[0] + 1) / 2))
    plt.subplot(2, 2, 4)
    plt.title(f"Noised Image 2 at t = {batch_t[1].item()}")
    plt.imshow(tensor_to_pil((batch_x_t[1] + 1) / 2))
    # plt.show()
    plt.savefig("imgs/forward_diffusion_original_vs_noised_images.png")
