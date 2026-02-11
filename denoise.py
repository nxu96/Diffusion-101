"""
denoise.py - Reverse (backward) denoising process for image generation.

This module implements the iterative reverse diffusion process that starts
from pure Gaussian noise and progressively removes noise to generate clean
images.  It supports optional LoRA weight merging for enhanced generation.

The reverse process uses the trained UNet to predict noise at each timestep,
then applies the DDPM posterior formula to compute the mean of x_{t-1} given
x_t, adding stochastic noise at every step except the last (t=0).

Usage:
    python denoise.py
"""

import matplotlib.pyplot as plt
import torch
from config import *
from dataset import tensor_to_pil
from diffusion import *
from lora import LoraLayer, inject_lora
from torch import nn


def backward_denoise(model, batch_x_t, batch_cls):
    """
    Perform the full reverse diffusion (denoising) process from timestep T-1
    down to 0, generating clean images from noise.

    At each timestep t, the model predicts the noise in x_t, and the DDPM
    posterior mean formula is used to compute x_{t-1}:

        mu_t = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * predicted_noise)

    For t > 0, Gaussian noise scaled by the posterior variance is added.
    For t = 0, the mean is returned directly (no noise).

    Args:
        model:     Trained UNet noise prediction model.
        batch_x_t: Initial noise tensor of shape (batch, 1, IMG_SIZE, IMG_SIZE).
        batch_cls: Class labels of shape (batch,), integers in [0, 9].

    Returns:
        steps: List of tensors tracking the denoising trajectory.
               steps[0] = initial noise, steps[-1] = final generated image.
               Length: T + 1.
    """
    # Store intermediate results for visualization of the denoising trajectory
    steps = [
        batch_x_t,
    ]

    # Access the precomputed noise schedule parameters from diffusion.py
    global alphas, alphas_cumprod, variance

    # Move all tensors and the model to the configured compute device
    model = model.to(DEVICE)
    batch_x_t = batch_x_t.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    variance = variance.to(DEVICE)
    batch_cls = batch_cls.to(DEVICE)

    # Switch to evaluation mode to disable batch normalization's running
    # statistics updates and use the learned population statistics instead.
    # This is critical because during inference, individual batch statistics
    # would be unreliable and inconsistent.
    model.eval()

    # Disable gradient computation for memory efficiency during inference
    with torch.no_grad():
        # Iterate backwards from T-1 down to 0 (reverse diffusion)
        for t in range(T - 1, -1, -1):
            # Create a batch of identical timestep values, e.g. [999, 999, ...]
            batch_t = torch.full((batch_x_t.size(0),), t).to(DEVICE)

            # Predict the noise component in x_t using the trained UNet
            batch_predict_noise_t = model(batch_x_t, batch_t, batch_cls)

            # --- Compute the posterior mean (mu_t) using the DDPM formula ---
            # Shape for broadcasting over (channel, height, width)
            shape = (batch_x_t.size(0), 1, 1, 1)

            # mu_t = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * noise)
            batch_mean_t = (
                1
                / torch.sqrt(alphas[batch_t].view(*shape))
                * (
                    batch_x_t
                    - (1 - alphas[batch_t].view(*shape))
                    / torch.sqrt(1 - alphas_cumprod[batch_t].view(*shape))
                    * batch_predict_noise_t
                )
            )

            if t != 0:
                # For t > 0: sample x_{t-1} = mu_t + sqrt(variance_t) * z
                # where z ~ N(0, I) is fresh Gaussian noise
                batch_x_t = batch_mean_t + torch.randn_like(batch_x_t) * torch.sqrt(
                    variance[batch_t].view(*shape)
                )
            else:
                # For t = 0: no noise is added; the mean IS the final output
                batch_x_t = batch_mean_t

            # Clamp pixel values to [-1, 1] to prevent accumulation of
            # numerical errors, and detach from the computation graph.
            batch_x_t = torch.clamp(batch_x_t, -1.0, 1.0).detach()

            # Record this denoising step for visualization
            steps.append(batch_x_t)

    return steps


# ===========================================================================
# Image generation demo (runs only when executed as a script)
# ===========================================================================
if __name__ == "__main__":
    # --- Load the pre-trained base model ---
    model = torch.load("model.pt", weights_only=False)

    # Flag to control whether LoRA weights are merged into the base model
    USE_LORA = True

    if USE_LORA:
        # --- Inject LoRA layers into cross-attention Q/K/V projections ---
        for name, layer in model.named_modules():
            name_cols = name.split(".")
            # Target only the query, key, and value projection layers
            filter_names = ["w_q", "w_k", "w_v"]
            if any(n in name_cols for n in filter_names) and isinstance(
                layer, nn.Linear
            ):
                inject_lora(model, name, layer)

        # --- Load saved LoRA weights ---
        try:
            restore_lora_state = torch.load("lora.pt", weights_only=False)
            model.load_state_dict(restore_lora_state, strict=False)
        except:
            raise Exception("LoRA weights not found")

        # Move model to compute device
        model = model.to(DEVICE)

        # --- Merge LoRA weights into the base model ---
        # Instead of keeping LoRA as a separate bypass during inference,
        # we fold the LoRA contribution directly into the base linear weight:
        #   W_merged = W_original + (A @ B) * (alpha / r)
        # This eliminates the LoRA overhead at inference time.
        # Collect LoRA layers first to avoid mutating module tree while iterating.
        lora_layers = [
            (name, layer)
            for name, layer in model.named_modules()
            if isinstance(layer, LoraLayer)
        ]
        for name, layer in lora_layers:
            name_cols = name.split(".")

            # Navigate to the parent module that contains this LoRA layer
            children = name_cols[:-1]
            cur_layer = model
            for child in children:
                cur_layer = getattr(cur_layer, child)

            # Compute the LoRA weight delta: (A @ B) * (alpha / r)
            lora_weight = (layer.lora_a @ layer.lora_b) * layer.alpha / layer.r

            # Add LoRA delta to the original linear weight (transposed because
            # nn.Linear stores weight as (out_features, in_features))
            layer.raw_linear.weight = nn.Parameter(
                layer.raw_linear.weight.add(lora_weight.T).to(DEVICE)
            )

            # Replace the LoRA wrapper with the merged linear layer
            setattr(cur_layer, name_cols[-1], layer.raw_linear)

    # Print the final model architecture
    print(model)

    # Print the number of parameters in the model.
    print(
        f"Number of parameters in the model: {sum(p.numel() for p in model.parameters()):,}"
    )

    # --- Generate images ---
    # Create a batch of pure Gaussian noise as the starting point
    batch_size = 10
    batch_x_t = torch.randn(size=(batch_size, 1, IMG_SIZE, IMG_SIZE))  # (10, 1, 48, 48)

    # Class labels for conditional generation: generate one image per digit (0-9)
    batch_cls = torch.arange(start=0, end=10, dtype=torch.long)

    # Run the reverse denoising process to generate images from noise
    steps = backward_denoise(model, batch_x_t, batch_cls)

    # --- Visualize the denoising trajectory ---
    num_imgs = 20  # Number of intermediate steps to display per digit

    plt.figure(figsize=(15, 15))
    for b in range(batch_size):
        for i in range(0, num_imgs):
            # Select evenly-spaced timesteps from the denoising trajectory
            idx = int(T / num_imgs) * (i + 1)
            # Rescale pixel values from [-1, 1] back to [0, 1] for display
            final_img = (steps[idx][b].to("cpu") + 1) / 2
            # Convert tensor to PIL image for matplotlib rendering
            final_img = tensor_to_pil(final_img)
            # Plot in a grid: rows = digits, columns = denoising steps
            plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
            plt.imshow(final_img)
    # plt.show()
    plt.savefig("imgs/conditional_inference.png")
