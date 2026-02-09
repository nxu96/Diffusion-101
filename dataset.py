"""
dataset.py - MNIST dataset loading and image transformation utilities.

This module sets up the MNIST training dataset with appropriate transforms
for converting PIL images to tensors (and back), resized to the configured
IMG_SIZE.  It also provides a tensor-to-PIL conversion pipeline used for
visualization during training and inference.
"""

import matplotlib.pyplot as plt
import torchvision
from config import *
from torchvision import transforms

# ---------------------------------------------------------------------------
# Transform: PIL image -> Tensor
# ---------------------------------------------------------------------------
# 1. Resize the original MNIST image (28x28) to (IMG_SIZE x IMG_SIZE).
# 2. Convert the PIL image to a float tensor with shape (C, H, W) and
#    pixel values normalized to the range [0, 1].
pil_to_tensor = transforms.Compose(
    [
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        ),  # Resize PIL image to uniform spatial dimensions
        transforms.ToTensor(),  # Convert PIL (H,W,C) -> Tensor (C,H,W), pixels in [0,1]
    ]
)

# ---------------------------------------------------------------------------
# Transform: Tensor -> PIL image
# ---------------------------------------------------------------------------
# 1. Scale pixel values from [0, 1] back to [0, 255].
# 2. Cast to uint8 (required by PIL).
# 3. Convert the tensor back to a PIL Image object for display/saving.
tensor_to_pil = transforms.Compose(
    [
        transforms.Lambda(
            lambda t: t * 255
        ),  # Rescale pixel values from [0,1] to [0,255]
        transforms.Lambda(
            lambda t: t.type(torch.uint8)
        ),  # Cast from FP32 to unsigned 8-bit integers for PIL compatibility
        transforms.ToPILImage(),  # Convert Tensor (C,H,W) -> PIL Image (H,W,C)
    ]
)

# ---------------------------------------------------------------------------
# MNIST Training Dataset
# ---------------------------------------------------------------------------
# Downloads the MNIST handwritten digits dataset (if not already present in
# the current directory) and applies the pil_to_tensor transform so that
# each sample is a (1, IMG_SIZE, IMG_SIZE) float tensor paired with its
# integer label (0-9).
train_dataset = torchvision.datasets.MNIST(
    root=".",  # Download / look for data in the current working directory
    train=True,  # Use the training split (60,000 images)
    download=True,  # Download if the dataset is not already cached locally
    transform=pil_to_tensor,  # Apply resize + to-tensor transform on each image
)

# ---------------------------------------------------------------------------
# Quick sanity-check visualization (runs only when executed as a script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Fetch the first sample from the training set
    img_tensor, label = train_dataset[0]
    print(f"Image label is {label}")
    print(f"Image shape is {img_tensor.shape}")
    print(f"Image type is {img_tensor.dtype}")
    print(f"Image device is {img_tensor.device}")
    print(f"Image value range is {img_tensor.min()} to {img_tensor.max()}")

    # Convert the tensor back to a PIL image and display it
    plt.figure(figsize=(5, 5))
    pil_img = tensor_to_pil(img_tensor)
    plt.imshow(pil_img)
    # plt.show()
    plt.savefig("imgs/mnist_sample.png")
