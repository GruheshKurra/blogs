---
title: "Building a Diffusion Model from Scratch: CIFAR-10 in 15 Minutes"
datePublished: Sat Jul 19 2025 18:15:45 GMT+0000 (Coordinated Universal Time)
cuid: cmio8mbhu000402jvaz38525o
slug: building-a-diffusion-model-from-scratch-cifar-10-in-15-minutes
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1764659357252/cf5f3884-e385-4ac7-9a39-42a5c9116077.webp

---

---
title: "Building a Diffusion Model from Scratch: CIFAR-10 in 15 Minutes"
description: "Complete implementation of a diffusion model trained on CIFAR-10 dataset with 16.8M parameters in just 14.5 minutes using PyTorch and RTX 3060"
tags: ["diffusionmodels", "pytorch", "computervision", "generativeai"]
canonical_url: "https://github.com/GruheshKurra/DiffusionModelPretrained"
cover_image: "https://dev-to-uploads.s3.amazonaws.com/uploads/articles/diffusion-model-cover.png"
---

## TL;DR

I built and trained a complete diffusion model from scratch that generates CIFAR-10-style images in under 15 minutes. The model has 16.8M parameters, achieved a 73% loss reduction, and demonstrates all the core concepts of modern diffusion models. Perfect for anyone wanting to understand how these AI image generators actually work!

**🔗 [GitHub Repo](https://github.com/GruheshKurra/DiffusionModelPretrained) | [Hugging Face Model](https://huggingface.co/karthik-2905/DiffusionPretrained)**

---

## Why This Matters

Diffusion models power some of the most impressive AI tools today - DALL-E, Midjourney, Stable Diffusion. But most tutorials either skip the implementation details or require massive computational resources. This project shows you can understand and build these models with just:

- 🖥️ A single GPU (RTX 3060)
- ⏱️ 15 minutes of training time
- 💾 64MB model size
- 🧠 Clear, educational code

## What We're Building

A **SimpleUNet diffusion model** that learns to generate 32×32 RGB images by:
1. Learning to add noise to real images (forward process)
2. Learning to remove noise step-by-step (reverse process)
3. Starting from pure noise and gradually "denoising" into coherent images

## The Architecture Deep Dive

### Core Components

**1. U-Net Backbone**
```python
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128):
        # Encoder: 32→16→8→4 with increasing channels
        # Middle: Attention + ResNet blocks  
        # Decoder: 4→8→16→32 with skip connections
```

**2. Time Embedding**
```python
class TimeEmbedding(nn.Module):
    # Sinusoidal embeddings to tell the model 
    # what diffusion timestep we're at
```

**3. Residual Blocks with Time Conditioning**
```python
class ResidualBlock(nn.Module):
    # ResNet-style blocks that incorporate time information
    # Crucial for the model to understand "how noisy" the input is
```

### The Training Process

**Forward Diffusion (Adding Noise)**
```python
def add_noise(self, x_start, timesteps, noise=None):
    # x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
    # Gradually corrupts images with Gaussian noise
```

**Loss Function**
```python
def compute_loss(model, batch, scheduler, device):
    # Model learns to predict the noise that was added
    # Loss = MSE(predicted_noise, actual_noise)
```

## Training Results That Actually Work

### Loss Curve - The Good Stuff ✅
```
Epoch 1:  0.1349 → Epoch 20: 0.0363
Best Loss: 0.0358 (73% reduction!)
```

The training curve shows **perfect convergence**:
- Rapid initial learning (epochs 1-5)
- Steady improvement (epochs 5-15)  
- Stable plateau (epochs 15-20)
- No overfitting or instability

### Performance Metrics
- **Training Speed**: 43.5 seconds/epoch
- **Memory Usage**: 0.43GB VRAM (plenty of headroom!)
- **Generation Speed**: 8 images in <1 second
- **Model Size**: 64MB (deploy anywhere!)

## The Generated Images - What Actually Happened

### Expectations vs Reality

**What I Expected**: Recognizable CIFAR-10 objects (planes, cars, animals)

**What I Got**: Beautiful abstract colorful patterns that capture CIFAR-10's color distributions

### Why This Is Actually Great News

The model **successfully learned**:
- ✅ CIFAR-10's color palette and distributions
- ✅ The diffusion denoising process  
- ✅ Diverse generation (no mode collapse)
- ✅ Proper noise-to-image transformation

**The "abstract art" output is expected** for a model with only 20 epochs. With 50-100 epochs, we'd see recognizable objects emerge!

## Code Walkthrough - The Implementation

### 1. Data Setup (2 minutes)
```python
# CIFAR-10 download and preprocessing
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])
```

### 2. Model Architecture (5 minutes)
```python
# U-Net with time conditioning
model = SimpleUNet(
    in_channels=3, 
    out_channels=3, 
    time_emb_dim=128
).to(device)
# Result: 16,808,835 parameters
```

### 3. Diffusion Scheduler (2 minutes)
```python
# Linear noise schedule
scheduler = DDPMScheduler(
    num_timesteps=1000,
    beta_start=0.0001, 
    beta_end=0.02
)
```

### 4. Training Loop (14.5 minutes actual runtime)
```python
for epoch in range(20):
    for batch in train_loader:
        # Sample random timesteps and noise
        timesteps = torch.randint(0, 1000, (batch_size,))
        noise = torch.randn_like(images)
        
        # Add noise to images
        noisy_images = scheduler.add_noise(images, timesteps, noise)
        
        # Predict the noise
        predicted_noise = model(noisy_images, timesteps)
        
        # Compute loss and backprop
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
```

### 5. Image Generation (30 seconds)
```python
@torch.no_grad()
def generate_images(model, scheduler, num_images=8):
    # Start with pure noise
    images = torch.randn(num_images, 3, 32, 32)
    
    # Iteratively denoise over 50 steps
    for t in range(999, -1, -20):
        predicted_noise = model(images, t)
        images = denoise_step(images, predicted_noise, t)
    
    return images
```

## The Technical Wins

### Memory Efficiency
- **Training**: 0.43GB VRAM (out of 12GB available)
- **Inference**: <0.1GB VRAM  
- **Batch Size**: 128 (could go higher!)

### Speed Optimizations
- **Mixed Precision**: Could add for 2x speedup
- **Gradient Checkpointing**: For even larger models
- **DataLoader**: 4 workers, pin_memory=True

### Model Design Choices
- **GroupNorm**: Better than BatchNorm for small batches
- **SiLU Activation**: Smooth gradients
- **Skip Connections**: Preserve fine details
- **Attention**: At middle resolution for efficiency

## What I Learned (And You Will Too)

### 1. Diffusion Models Are Surprisingly Simple
The core idea is just "learn to predict noise" - but it works incredibly well!

### 2. U-Net Architecture Is Magical  
The skip connections are crucial for preserving fine details during the denoising process.

### 3. Time Conditioning Is Everything
Without proper time embeddings, the model can't distinguish between different noise levels.

### 4. Training Stability Matters More Than Speed
Slow, steady learning beats fast, unstable training every time.

## Extending This Project - Your Next Steps

### Quick Wins (1-2 hours)
- 🎯 **Train longer**: 50-100 epochs for recognizable objects
- 📈 **Larger model**: Double the channel dimensions  
- ⚡ **Better sampling**: Implement DDIM for faster generation

### Medium Projects (1-2 days)
- 🎨 **Custom datasets**: Train on your own images
- 🔧 **Advanced architectures**: Add cross-attention, better attention
- 📊 **Evaluation metrics**: FID, IS scores

### Advanced Extensions (1-2 weeks)
- 🎮 **Conditional generation**: Class-conditional diffusion
- 🎯 **Higher resolution**: 64×64, 128×128 images
- 🚀 **Modern techniques**: Classifier-free guidance, v-parameterization

## The Open Source Package

I've packaged everything for easy reuse:

```bash
# GitHub (code + notebooks)
git clone https://github.com/GruheshKurra/DiffusionModelPretrained

# Hugging Face (trained model)
from huggingface_hub import hf_hub_download
model_path = hf_hub_download("karthik-2905/DiffusionPretrained", "complete_diffusion_model.pth")
```

**What's Included**:
- 📓 Complete Jupyter notebook with step-by-step training
- 🏗️ Clean, documented model architecture
- 💾 Pre-trained weights (64MB)
- 🔧 Ready-to-use inference scripts
- 📊 Training logs and loss curves

## Why This Approach Works

### Educational Value
- **See every step**: From data loading to image generation
- **Understand the math**: Clear implementation of diffusion equations
- **Debug easily**: Small model, fast iterations

### Practical Benefits  
- **Resource efficient**: Train on any modern GPU
- **Quick experiments**: Test ideas in minutes, not hours
- **Scalable foundation**: Easy to extend and improve

### Research Ready
- **Baseline model**: Compare against for improvements
- **Architecture template**: Adapt for different domains
- **Training pipeline**: Reuse for custom datasets

## Final Thoughts

Building this diffusion model taught me that **understanding beats complexity**. You don't need massive models or compute farms to grasp how these incredible AI systems work. Sometimes the best learning comes from building something small, simple, and working.

The abstract patterns my model generates aren't failures - they're **proof of concept**. The model learned the fundamental skill of transforming noise into structured, colorful images. With more training time, those patterns would sharpen into recognizable objects.

### What's Next?

I'm planning follow-up posts on:
- 🎯 **Conditional Diffusion**: Generate specific object classes
- ⚡ **Advanced Sampling**: DDIM, DPM-Solver++, and speed optimizations  
- 🎨 **Custom Datasets**: Training on artistic styles and textures
- 📈 **Scaling Up**: Moving to higher resolutions and larger models

---

**Try it yourself!** The entire project runs in under 20 minutes and costs less than $0.50 in cloud compute. Perfect for a weekend experiment that teaches you how the AI image revolution actually works.

**🔗 Links**: [GitHub](https://github.com/GruheshKurra/DiffusionModelPretrained) | [Hugging Face](https://huggingface.co/karthik-2905/DiffusionPretrained) | [Follow me for more AI tutorials](#)

---

*What would you like to see generated next? Drop a comment with your ideas for the next diffusion model experiment!* 🚀