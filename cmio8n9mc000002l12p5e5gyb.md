---
title: "Building a GAN from Scratch: My Journey into Generative AI 🤖"
datePublished: Sun Jul 13 2025 06:10:59 GMT+0000 (Coordinated Universal Time)
cuid: cmio8n9mc000002l12p5e5gyb
slug: building-a-gan-from-scratch-my-journey-into-generative-ai

---


*How I implemented Generative Adversarial Networks to generate MNIST digits and what I learned along the way*

## TL;DR 🚀

I built a complete GAN implementation from scratch using PyTorch that generates realistic MNIST handwritten digits. The project includes both standard and optimized versions, comprehensive logging, and supports multiple devices (MPS, CUDA, CPU). 

**Links:**
- [GitHub Repository](https://github.com/GruheshKurra/GAN_Implementation)
- [Hugging Face Space](https://huggingface.co/karthik-2905/GAN_Implementation)

## The Challenge 💡

Generative Adversarial Networks (GANs) have always fascinated me - the idea of two neural networks competing against each other to create realistic data seemed like something out of science fiction. So I decided to dive deep and build one from scratch.

**What I wanted to achieve:**
- Generate realistic handwritten digits
- Understand the adversarial training process
- Create both standard and optimized versions
- Make it work across different hardware (Apple Silicon, NVIDIA GPUs, CPU)

## The Architecture 🏗️

### Generator Network

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, image_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)
```

### Discriminator Network

```python
class Discriminator(nn.Module):
    def __init__(self, image_dim=784, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, x):
        return self.model(x)
```

## Key Implementation Details 🔧

### 1. Adversarial Training Loop

The magic happens in the training loop where both networks compete:

```python
# Train Discriminator
real_loss = criterion(discriminator(real_images), real_labels)
fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
d_loss = real_loss + fake_loss

# Train Generator
g_loss = criterion(discriminator(fake_images), real_labels)  # Trick discriminator
```

### 2. Device Optimization

I implemented automatic device detection for maximum compatibility:

```python
def _setup_device(self, device: str) -> torch.device:
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')  # Apple Silicon
        else:
            return torch.device('cpu')
```

### 3. Two Training Modes

**Standard Mode**: Full dataset, high quality
- 60K samples, 100D latent space
- ~30 minutes training time
- 3.5M generator parameters

**Lite Mode**: Fast experimentation
- 10K samples, 64D latent space
- ~5 minutes training time
- 576K generator parameters

## Results & Performance 📊

| Mode | Training Time | Generator Loss | Discriminator Loss | Quality |
|------|---------------|----------------|-------------------|---------|
| Standard | ~30 min | ~1.5 | ~0.7 | High |
| Lite | ~5 min | ~2.0 | ~0.6 | Good |

The generated digits look surprisingly realistic! Here's what the training progression looks like:

```
Epoch [1/50] D Loss: 1.414 G Loss: 0.727
Epoch [10/50] D Loss: 0.687 G Loss: 1.892
Epoch [25/50] D Loss: 0.654 G Loss: 1.456
Epoch [50/50] D Loss: 0.623 G Loss: 1.234
```

## Challenges I Faced 😅

### 1. Mode Collapse
Early versions would generate the same digit repeatedly. Fixed with:
- Better weight initialization
- Balanced training between G and D
- Proper learning rates

### 2. Training Instability
GANs are notoriously hard to train. Solutions:
- Adam optimizer with β₁=0.5, β₂=0.999
- LeakyReLU in discriminator
- Batch normalization in generator

### 3. Memory Management
Training on Apple Silicon required special handling:

```python
if self.device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
    torch.mps.empty_cache()
```

## What I Learned 🎓

1. **GANs are an art form** - Getting them to work requires patience and experimentation
2. **Logging is crucial** - Comprehensive logging helped debug training issues
3. **Hardware optimization matters** - Supporting multiple devices makes the project accessible
4. **Code organization** - Clean, modular code makes experimentation easier

## Technical Highlights 🌟

- **Comprehensive logging** with real-time progress tracking
- **Automatic device detection** (MPS/CUDA/CPU)
- **Model persistence** for saving/loading trained models
- **Visualization tools** for monitoring training progress
- **Memory optimization** for efficient training
- **Jupyter notebook** for interactive experimentation

## Future Improvements 🔮

- [ ] Implement DCGAN with convolutional layers
- [ ] Add support for colored images (CIFAR-10)
- [ ] Conditional GAN for digit-specific generation
- [ ] Web interface for interactive generation
- [ ] More advanced architectures (StyleGAN, Progressive GAN)

## Try It Yourself! 🛠️

```bash
# Clone the repository
git clone https://github.com/GruheshKurra/GAN_Implementation.git
cd GAN_Implementation

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Gan.ipynb
```

## Resources 📚

- **GitHub Repository**: [GAN_Implementation](https://github.com/GruheshKurra/GAN_Implementation)
- **Hugging Face**: [karthik-2905/GAN_Implementation](https://huggingface.co/karthik-2905/GAN_Implementation)
- **Original GAN Paper**: [Goodfellow et al. (2014)](https://arxiv.org/abs/1406.2661)

## Final Thoughts 💭

Building a GAN from scratch was both challenging and rewarding. It gave me deep insights into:
- How adversarial training works in practice
- The importance of proper architecture design
- Hardware optimization for deep learning
- The art of debugging neural networks

The most satisfying moment was seeing those first realistic digits emerge from random noise - it felt like digital magic! ✨

---

**What's your experience with GANs? Have you tried building one from scratch? Let me know in the comments!** 