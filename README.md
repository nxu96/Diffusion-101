# Diffusion-101

A from-scratch implementation of a **class-conditioned DDPM** (Denoising Diffusion Probabilistic Model) trained on the MNIST handwritten digits dataset. Inspired by the Stable Diffusion / Latent Diffusion architecture, the model uses **cross-attention conditioning** to enable controlled generation — you specify a target digit label (0-9) and the model generates the corresponding digit image. This is the same conditioning mechanism that Stable Diffusion uses with text prompts, applied here with discrete class labels.

## Model Output

Class-guided generation of 10 images, one for each digit 0-9:

![Model Output](imgs/stable-inference.png)

## Architecture

The project implements the full diffusion pipeline with the following components:

### Forward Diffusion (`diffusion.py`)

Implements the forward noising process that gradually adds Gaussian noise to clean images over T=1000 timesteps using a linear beta schedule. Uses the DDPM closed-form formula to jump directly from x_0 to any x_t in a single step:

```
x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
```

### UNet Noise Predictor (`unet.py`)

A standard UNet encoder-decoder architecture with skip connections that predicts the noise added at each timestep. Channel progression: 1 → 64 → 128 → 256 → 512 → 1024.

- **Encoder:** ConvBlock → MaxPool downsampling at each level
- **Bottleneck:** ConvBlock at the deepest level (1024 channels)
- **Decoder:** ConvTranspose2d upsampling → concatenate skip connection → ConvBlock
- **Output:** 1×1 Conv2d to restore the original channel count

### Convolutional Block (`conv_block.py`)

The fundamental building block of the UNet. Each block applies:

1. Conv2d → BatchNorm → ReLU (changes channel count)
2. Timestep embedding injection (additive, broadcast over spatial dims)
3. Conv2d → BatchNorm → ReLU (preserves channel count)
4. Cross-attention with the class embedding

### Timestep Embedding (`time_position_emb.py`)

Sinusoidal positional encoding (from "Attention Is All You Need") adapted for diffusion timesteps. Encodes integer timesteps into continuous 256-dimensional vectors using interleaved sine and cosine functions at geometrically spaced frequencies.

### Cross-Attention (`cross_attn.py`)

Class-conditioning module where image feature pixels serve as queries and the class embedding (digit label 0-9) serves as key/value. Follows the standard Transformer pattern: Attention → Residual + LayerNorm → FeedForward → Residual + LayerNorm.

### Reverse Denoising (`denoise.py`)

Iterative reverse diffusion process that starts from pure Gaussian noise and progressively denoises over T steps using the trained UNet. At each step, the DDPM posterior mean formula is applied:

```
μ_t = (1/√α_t) * (x_t - (1-α_t)/√(1-α̅_t) * predicted_noise)
```

### LoRA Fine-Tuning (`lora.py`, `lora_finetune.py`)

Low-Rank Adaptation (LoRA) support for parameter-efficient fine-tuning. LoRA layers are injected into the cross-attention Q/K/V projections, keeping the base model frozen while training only the low-rank matrices A and B. At inference time, LoRA weights can be merged back into the base model for zero overhead.

### Dataset (`dataset.py`)

MNIST handwritten digits (60,000 training images), resized from 28×28 to 48×48. Pixel values are normalized to [-1, 1] to match the Gaussian noise distribution.

## Configuration (`config.py`)

| Parameter    | Value | Description                         |
|------------- |-------|-------------------------------------|
| `IMG_SIZE`   | 48    | Input image resolution (48×48)      |
| `T`          | 1000  | Number of diffusion timesteps       |
| `LORA_ALPHA` | 1     | LoRA scaling factor                 |
| `LORA_R`     | 8     | LoRA rank                           |
| `DEVICE`     | auto  | Uses CUDA if available, else CPU    |

## Usage

### Train the base model

```bash
python train.py
```

Trains the UNet for 200 epochs with batch size 400, L1 loss, and Adam optimizer (lr=0.001). Checkpoints are saved to `model.pt` after each epoch. Training metrics are logged to TensorBoard.

### LoRA fine-tuning

```bash
python lora_finetune.py
```

Fine-tunes the pre-trained model by injecting LoRA into cross-attention layers. Only LoRA weights are saved to `lora.pt`.

### Generate images

```bash
python denoise.py
```

Generates digit images (0-9) by running the reverse denoising process from pure noise, optionally merging LoRA weights.

## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
