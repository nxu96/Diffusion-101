"""
train.py - Main training script for the diffusion model.

Trains a UNet to predict the noise added during the forward diffusion process
on the MNIST dataset.  The model learns to denoise images conditioned on the
digit class (0-9).  Training uses L1 loss (mean absolute error) between the
predicted noise and the actual Gaussian noise, with the Adam optimizer.

Model checkpoints are saved atomically after each epoch to 'model.pt'.
Training metrics (loss) are logged to TensorBoard.

Usage:
    python train.py
"""

import os

import torch
from config import *
from dataset import train_dataset
from diffusion import forward_diffusion
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unet import UNet

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPOCH = 200  # Total number of training epochs
BATCH_SIZE = 1000  # Number of images per mini-batch

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
# Wraps the MNIST training dataset in a DataLoader for batched, shuffled,
# multi-worker loading.  persistent_workers=True keeps worker processes alive
# between epochs to avoid re-initialization overhead.
dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    persistent_workers=True,
    shuffle=True,
)

# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
# Attempt to load a previously saved model checkpoint.  If no checkpoint
# exists, create a fresh UNet with 1 input channel (grayscale).
try:
    model = torch.load("model.pt", weights_only=False)
except:
    model = UNet(img_channel=1).to(DEVICE)  # UNet for single-channel (grayscale) images

# ---------------------------------------------------------------------------
# Optimizer and loss function
# ---------------------------------------------------------------------------
# Adam optimizer with learning rate 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# L1 loss (mean absolute error) between predicted and actual noise
loss_fn = nn.L1Loss()

# TensorBoard writer for logging training metrics
writer = SummaryWriter()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Set model to training mode (enables dropout, batch norm updates, etc.)
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

            # Move class labels (digit IDs 0-9) to the compute device.
            # These are used for class-conditioned generation via cross-attention.
            batch_cls = batch_cls.to(DEVICE)

            # Sample a random timestep for each image in the batch.
            # Each t is in [0, T), determining how much noise is added.
            batch_t = torch.randint(0, T, (batch_x.size(0),)).to(DEVICE)

            # Apply forward diffusion: produce noised images x_t and the
            # ground-truth noise that was added at each timestep.
            batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)

            # Feed the noised images, timesteps, and class labels into the
            # UNet to predict the noise component.
            batch_predict_t = model(batch_x_t, batch_t, batch_cls)

            # Compute L1 loss between predicted noise and actual noise
            loss = loss_fn(batch_predict_t, batch_noise_t)

            # Backpropagation: zero gradients, compute gradients, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss value for this batch
            last_loss = loss.item()
            # Log the training loss to TensorBoard
            writer.add_scalar("Loss/train", last_loss, n_iter)
            n_iter += 1

        # Print epoch summary with the last batch's loss
        print("epoch:{} loss={}".format(epoch, last_loss))

        # Save model checkpoint atomically: write to a temporary file first,
        # then rename.  This prevents corruption if the process is interrupted
        # during the save.
        torch.save(model, "model.pt.tmp")
        os.replace("model.pt.tmp", "model.pt")
    print("Training complete!")
    writer.close()
