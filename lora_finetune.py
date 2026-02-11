"""
lora_finetune.py - LoRA (Low-Rank Adaptation) fine-tuning script.

This script fine-tunes a pre-trained diffusion UNet using LoRA by:
  1. Loading the pre-trained base model from 'model.pt'
  2. Injecting LoRA layers into the cross-attention Q/K/V linear projections
  3. Freezing all base model parameters (only LoRA weights are trainable)
  4. Training only the LoRA parameters on the MNIST dataset
  5. Saving only the LoRA weights to 'lora.pt' after each epoch

This approach is parameter-efficient: only a small number of low-rank
matrices are trained while the vast majority of the model stays frozen.

Usage:
    python lora_finetune.py
"""

import os

import torch
from config import *
from dataset import train_dataset
from diffusion import forward_diffusion
from lora import inject_lora
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unet import UNet

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCH = 200  # Total number of fine-tuning epochs
BATCH_SIZE = 400  # Number of images per mini-batch

# ---------------------------------------------------------------------------
# Main fine-tuning script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Load the pre-trained base model ---
    model = torch.load("model.pt", weights_only=False)

    # --- Inject LoRA into cross-attention linear layers ---
    # Iterate over all named modules in the model and find the Q, K, V
    # linear projections used in cross-attention blocks.
    for name, layer in model.named_modules():
        name_cols = name.split(".")
        # Target only the query, key, and value projection layers
        filter_names = ["w_q", "w_k", "w_v"]
        # Check if this module is one of the target linear layers
        if any(n in name_cols for n in filter_names) and isinstance(layer, nn.Linear):
            # Replace the nn.Linear with a LoraLayer wrapper
            inject_lora(model, name, layer)

    # --- Restore previously saved LoRA weights (if available) ---
    # This allows resuming fine-tuning from a prior checkpoint.
    # strict=False allows loading only the LoRA parameters while ignoring
    # the rest of the model state dict.
    try:
        restore_lora_state = torch.load("lora.pt", weights_only=False)
        model.load_state_dict(restore_lora_state, strict=False)
    except:
        pass

    # Move the model (with injected LoRA layers) to the compute device
    model = model.to(DEVICE)

    # --- Freeze all non-LoRA parameters ---
    # Only the lora_a and lora_b matrices will receive gradient updates;
    # all other parameters (base model weights) remain frozen.
    for name, param in model.named_parameters():
        if name.split(".")[-1] not in ["lora_a", "lora_b"]:
            # NOTE: This is how we freeze the base model parameters.
            # Freeze base model parameters by disabling gradient computation
            param.requires_grad = False
        else:
            # Enable gradient computation for LoRA parameters
            param.requires_grad = True

    # --- Data loader ---
    # Same setup as the main training script
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        shuffle=True,
    )

    # --- Optimizer ---
    # Only update parameters that require gradients (i.e., LoRA weights).
    # filter() ensures the optimizer only tracks LoRA parameters.
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad == True, model.parameters()), lr=0.001
    )

    # L1 loss (mean absolute error) between predicted and actual noise
    loss_fn = nn.L1Loss()

    # Print the model architecture to verify LoRA injection
    print(model)

    # TensorBoard writer for logging training metrics
    writer = SummaryWriter()

    # --- Fine-tuning loop ---
    model.train()
    # Global iteration counter for TensorBoard logging
    n_iter = 0

    for epoch in range(EPOCH):
        # Track the loss of the last batch in each epoch for reporting
        last_loss = 0

        for batch_x, batch_cls in dataloader:
            # Rescale pixel values from [0, 1] to [-1, 1] to match the
            # Gaussian noise distribution used in the diffusion process.
            batch_x = batch_x.to(DEVICE) * 2 - 1

            # Move class labels (digit IDs 0-9) to the compute device
            batch_cls = batch_cls.to(DEVICE)

            # Sample a random timestep for each image in the batch
            batch_t = torch.randint(0, T, (batch_x.size(0),)).to(DEVICE)

            # Apply forward diffusion to produce noised images and ground-truth noise
            batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)

            # Predict the noise using the model (LoRA-augmented UNet)
            batch_predict_t = model(batch_x_t, batch_t, batch_cls)

            # Compute L1 loss between predicted noise and actual noise
            loss = loss_fn(batch_predict_t, batch_noise_t)

            # Backpropagation: zero gradients, compute gradients, update LoRA weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss value for this batch
            last_loss = loss.item()
            # Log the training loss to TensorBoard
            writer.add_scalar("Loss/train", last_loss, n_iter)
            n_iter += 1

        # Print epoch summary
        print("epoch:{} loss={}".format(epoch, last_loss))

        # --- Save only LoRA weights ---
        # Extract only the lora_a and lora_b parameters from the model
        # state dict, keeping the checkpoint small and focused.
        lora_state = {}
        for name, param in model.named_parameters():
            name_cols = name.split(".")
            filter_names = ["lora_a", "lora_b"]
            # Check if this parameter is a LoRA weight
            if any(n == name_cols[-1] for n in filter_names):
                lora_state[name] = param

        # Save atomically: write to temp file, then rename to prevent corruption
        torch.save(lora_state, "lora.pt.tmp")
        os.replace("lora.pt.tmp", "lora.pt")
