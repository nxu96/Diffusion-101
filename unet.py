"""
unet.py - UNet architecture for noise prediction in diffusion models.

Implements a standard UNet with:
  - Encoder path: progressively doubles the channel count and halves spatial size
  - Bottleneck: highest channel count, smallest spatial resolution
  - Decoder path: progressively halves channels and doubles spatial size
  - Skip connections (residual): concatenate encoder features with decoder features

Each convolutional block (ConvBlock) incorporates:
  - Diffusion timestep conditioning via additive time embeddings
  - Class conditioning via cross-attention with the class embedding

The network predicts the noise epsilon added during the forward diffusion
process, which is used as the training target in DDPM.
"""

import torch
from config import *
from conv_block import ConvBlock
from dataset import train_dataset
from diffusion import forward_diffusion
from time_position_emb import TimePositionEmbedding
from torch import nn


class UNet(nn.Module):
    """
    UNet noise prediction network for class-conditioned diffusion.

    Architecture overview:
      Input (batch, img_channel, H, W)
        -> Encoder: [ConvBlock -> MaxPool] x (n_levels - 1) -> ConvBlock (bottleneck)
        -> Decoder: [ConvTranspose -> Concat(skip) -> ConvBlock] x (n_levels - 1)
        -> 1x1 Conv to restore original channel count
      Output (batch, img_channel, H, W)  -- predicted noise

    Args:
        img_channel  (int):       Number of image channels (1 for grayscale MNIST).
        channels     (list[int]): Channel sizes for each encoder/decoder level.
                                  Default: [64, 128, 256, 512, 1024].
        time_emb_size(int):       Dimension of the timestep embedding. Default: 256.
        qsize        (int):       Query/key size for cross-attention. Default: 16.
        vsize        (int):       Value size for cross-attention. Default: 16.
        fsize        (int):       Feed-forward hidden size in cross-attention. Default: 32.
        cls_emb_size (int):       Dimension of the class embedding. Default: 32.
    """

    def __init__(
        self,
        img_channel,
        channels=[64, 128, 256, 512, 1024],
        time_emb_size=256,
        qsize=16,
        vsize=16,
        fsize=32,
        cls_emb_size=32,
    ):
        super().__init__()

        # Prepend the image channel count to the channel list so that the
        # first ConvBlock maps from img_channel -> channels[0].
        # [1, 64, 128, 256, 512, 1024]
        channels = [img_channel] + channels

        # --- Timestep embedding ---
        # Converts integer timestep t into a continuous vector of size time_emb_size
        # using sinusoidal positional encoding followed by a linear + ReLU layer.
        # NOTE: The projection and activation here help the model learn a general
        # and better internal representation of time that all blocks can benefit from.
        self.time_emb = nn.Sequential(
            TimePositionEmbedding(
                time_emb_size
            ),  # Sinusoidal encoding: int -> (time_emb_size,)
            nn.Linear(time_emb_size, time_emb_size),  # Learnable projection
            nn.ReLU(),  # Non-linearity
        )

        # --- Class embedding ---
        # Learnable lookup table mapping class labels (0-9) to dense vectors
        # of size cls_emb_size.  Fed into cross-attention as key/value.
        self.cls_emb = nn.Embedding(10, cls_emb_size)

        # --- Encoder convolutional blocks ---
        # Each block transforms channels[i] -> channels[i+1].
        # The last encoder block serves as the bottleneck (no pooling after it).
        self.enc_convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_convs.append(
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    time_emb_size,
                    qsize,
                    vsize,
                    fsize,
                    cls_emb_size,
                )
            )

        # --- Max-pooling layers (encoder downsampling) ---
        # Applied after every encoder ConvBlock except the last one (bottleneck).
        # Each pool halves the spatial dimensions (H, W).
        self.maxpools = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.maxpools.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # --- Transposed convolution layers (decoder upsampling) ---
        # Each deconv doubles the spatial dimensions and halves the channel count,
        # preparing features for concatenation with the corresponding skip connection.
        self.deconvs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.deconvs.append(
                nn.ConvTranspose2d(
                    channels[-i - 1], channels[-i - 2], kernel_size=2, stride=2
                )
            )

        # --- Decoder convolutional blocks ---
        # Each block receives the concatenation of the upsampled features and
        # the skip-connection features (hence input channels = 2 * channels),
        # and outputs channels[-i-2].
        self.dec_convs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.dec_convs.append(
                ConvBlock(
                    channels[-i - 1],
                    channels[-i - 2],
                    time_emb_size,
                    qsize,
                    vsize,
                    fsize,
                    cls_emb_size,
                )
            )

        # --- Output projection ---
        # 1x1 convolution that maps from channels[1] back to img_channel,
        # producing the final noise prediction with the same shape as the input.
        self.output = nn.Conv2d(
            channels[1], img_channel, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, t, cls):
        """
        Forward pass: predict the noise added at timestep t, conditioned on class cls.

        Args:
            x:   Noised image tensor of shape (batch, img_channel, H, W).
            t:   Timestep indices of shape (batch,).
            cls: Class labels of shape (batch,), integers in [0, 9].

        Returns:
            Predicted noise tensor of shape (batch, img_channel, H, W).
        """
        # --- Compute embeddings ---
        # Convert timestep integers to continuous embeddings
        t_emb = self.time_emb(t)  # (batch, time_emb_size)
        # Convert class labels to dense embedding vectors
        cls_emb = self.cls_emb(cls)  # (batch, cls_emb_size)

        # --- Encoder path ---
        residual = []  # Store intermediate outputs for skip connections
        for i, conv in enumerate(self.enc_convs):
            # Apply convolutional block with time and class conditioning
            x = conv(x, t_emb, cls_emb)
            if i != len(self.enc_convs) - 1:
                # Save output for skip connection before downsampling
                residual.append(x)
                # Downsample spatial dimensions by 2x
                x = self.maxpools[i](x)

        # --- Decoder path ---
        for i, deconv in enumerate(self.deconvs):
            # Upsample spatial dimensions by 2x and halve channels
            x = deconv(x)
            # Pop the most recent skip-connection features (LIFO order)
            residual_x = residual.pop(-1)
            # Concatenate skip features with upsampled features along the
            # channel dimension and apply the decoder ConvBlock
            x = self.dec_convs[i](torch.cat((residual_x, x), dim=1), t_emb, cls_emb)

        # Apply 1x1 convolution to restore the original channel count
        return self.output(x)


# ===========================================================================
# Quick sanity check (runs only when executed as a script)
# ===========================================================================
if __name__ == "__main__":
    # Build a mini-batch of 2 images from the training set
    batch_x = torch.stack((train_dataset[0][0], train_dataset[1][0]), dim=0).to(
        DEVICE
    )  # (2, 1, 48, 48)
    # Rescale pixel values from [0, 1] to [-1, 1] to match Gaussian noise range
    batch_x = batch_x * 2 - 1
    # Create class label tensor for the two images
    batch_cls = torch.tensor(
        [train_dataset[0][1], train_dataset[1][1]], dtype=torch.long
    ).to(DEVICE)

    # Generate random diffusion timesteps for each image
    batch_t = torch.randint(0, T, size=(batch_x.size(0),)).to(DEVICE)
    # Apply forward diffusion to get noised images and ground-truth noise
    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)

    print("batch_x_t:", batch_x_t.size())  # Expected: (2, 1, 48, 48)
    print("batch_noise_t:", batch_noise_t.size())  # Expected: (2, 1, 48, 48)

    # Instantiate the UNet and predict noise
    unet = UNet(img_channel=1).to(DEVICE)
    batch_predict_noise_t = unet(batch_x_t, batch_t, batch_cls)
    print(
        "batch_predict_noise_t:", batch_predict_noise_t.size()
    )  # Expected: (2, 1, 48, 48)
