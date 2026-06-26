# Subtask 2 — Class-Conditional Image Generation with Diffusion Models

## Overview


It implements a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** for class-conditional image generation on **CIFAR-10**, using **PyTorch** and the **HuggingFace Diffusers** library. the model fine-tunes a publicly available pre-trained unconditional DDPM (`google/ddpm-cifar10-32`) by patching it with a class embedding layer. It tries to implement class-specific image generation with **Classifier-Free Guidance (CFG)**.

---

## Repository Structure

```text
Task2/Subtask2/
├── data/                             # CIFAR-10 dataset (auto-downloaded)
│   ├── cifar-10-batches-py/          # Extracted CIFAR-10 data
│   └── cifar-10-python.tar.gz        # Compressed CIFAR-10 archive
├── model_weights.pth                 # Saved model weights after training (≈143 MB)
├── task2-subtask2-pt (2).ipynb       # Main notebook: model, training, generation
└── README.md                         # This file
```

---

## Usage Instructions

### Training & Generation
The full pipeline is contained in `task2-subtask2-pt (2).ipynb`. Run the notebook cells sequentially to:

1. **Define the model** — Patch a pre-trained unconditional U-Net with a learned class embedding.
2. **Prepare data** — Load CIFAR-10 with train/val/test splits and diffusion-standard normalization.
3. **Fine-tune for 10 epochs** — Train the conditional DDPM with CFG dropout.
4. **Generate images** — Produce class-specific 32×32 images via reverse diffusion with guidance.

---

## Model Architecture

### `ConditionalDDPM`

The model is built by adding a class embedding layer to a pre-trained unconditional `UNet2DModel` (from HuggingFace's `diffusers`) to support class-conditional generation:

```python

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes=10, model_id="google/ddpm-cifar10-32"):
        super().__init__()
        
        #Load pre-trained model
        self.unet = UNet2DModel.from_pretrained(model_id)
        
        # Extract the exact time embedding dimension (same as class embedding)
     
        time_embed_dim = self.unet.time_embedding.linear_2.out_features
        # add the class embedding layer
        self.unet.class_embedding = nn.Embedding(num_classes, time_embed_dim)
        
        nn.init.normal_(self.unet.class_embedding.weight, std=0.02)

    def forward(self, sample, timestep, class_labels):
        return self.unet(sample, timestep, class_labels=class_labels).sample


```

**Key Design Decisions:**

| Decision | Rationale |
| --- | --- |
| **Patching vs. building from scratch** | The pre-trained `google/ddpm-cifar10-32` already has strong image priors for CIFAR-10. Patching it with a class embedding avoids training from scratch and leverages transfer learning. |
| **`num_classes = 11`** | 10 CIFAR-10 classes (0–9) + 1 **null token** (class 10) used for unconditional guidance during CFG. |
| **Using torch.compile** | Used torch.compile for an approximate 20% faster model training, and only a slight overhead. also used float32 matmul cores present in RTX5050 for further gain in speed. |
| **Disabling CUDNN** | CUDNN is not supported on RTX 5050 (a Mobile GPU). The notebook runs successfully on the available hardware by leveraging PyTorch's backend flexibility (MPS/CUDA). |
|**Differential learning rates**|To prevent destroying the weights of the U-net, it's learning rate is set very low, while the class embedding is set to 1e-3. To prevent the time embeddings from sabotaging the class embedding, it's learning rate was also increased to 1e-4|
| **Choosing the Guidance scale (CFG)**| Found 3.0 through trial and error. |




---

## Data Pipeline

### CIFAR-10 Splits

```python
def get_cifar10_splits(batch_size=64, val_split=0.1, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Maps [0,1] → [-1,1]
    ])
    
    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    val_size = int(len(full_train) * val_split)  # 5,000
    train_size = len(full_train) - val_size       # 45,000
    
    train, val = random_split(full_train, [train_size, val_size], 
                              generator=torch.Generator().manual_seed(seed))
```

| Split | Size | Batches (bs=64) |
| --- | --- | --- |
| **Train** | ~45,000 | 703 |
| **Validation** | ~5,000 | 79 |
| **Test** | 10,000 | 157 |

**Normalization:** Images are normalized to `[-1, 1]` — the standard diffusion normalization range. This is critical because the DDPM noise scheduler assumes inputs in this range.

---

## Training: Conditional DDPM with CFG Dropout

### Training Loop

The training loop implements the standard DDPM training objective with an additional **Classifier-Free Guidance (CFG) dropout** mechanism:

```
For each batch:
    1. Sample random timesteps t ~ Uniform(0, T)
    2. Sample Gaussian noise ε ~ N(0, I)
    3. Corrupt clean images: x_t = scheduler.add_noise(x_0, ε, t)
    4. CFG Dropout: With probability p=0.1, replace class labels with null token (10)
    5. Predict noise: ε̂ = model(x_t, t, class_labels)
    6. Loss = MSE(ε̂, ε)
    7. Backpropagate and update weights
```

### Classifier-Free Guidance (CFG) — Training

During training, **10% of the time**, the true class label is replaced with a **null token** (class index 10). This trains the model to produce both:
- **Conditional predictions** (given a real class label)
- **Unconditional predictions** (given the null token)

This dual capability is essential for CFG-guided generation at inference time.

### Training Configuration

| Parameter | Value |
| --- | --- |
| **Framework** | PyTorch + HuggingFace Diffusers |
| **Pre-trained Model** | `google/ddpm-cifar10-32` |
| **Optimizer** | AdamW (lr = 4e-5) |
| **Epochs** | 10 |
| **Batch Size** | 64 |
| **Image Size** | 32 × 32 × 3 |
| **Noise Scheduler** | DDPMScheduler (1000 timesteps) |
| **CFG Dropout Rate** | p = 0.1 (10% unconditional) |
| **Null Class Token** | Index 10 (11th embedding) |
| **Loss** | MSE (noise prediction) |
| **Accelerator** | GPU (CUDA) |
| **Total Training Time** | ~64.7 minutes (3882.58 seconds) |

### Training Results

The model was trained for 10 epochs on the training split:

| Epoch | Duration | Status |
| --- | --- | --- |
| 1–10 | ~6.5 min/epoch | Completed successfully |

Total training duration: **3,882.58 seconds** (≈ 64.7 minutes).

After training, model weights are saved to `model_weights.pth` (≈143 MB).

---

## Inference: CFG-Guided Generation

### Reverse Diffusion with Classifier-Free Guidance

At inference time, the model generates class-specific images using the **CFG extrapolation formula**:

```
ε̂_guided = ε̂_uncond + w × (ε̂_cond - ε̂_uncond)
```

Where:
- `ε̂_cond` = noise prediction given the target class label
- `ε̂_uncond` = noise prediction given the null token (class 10)
- `w` = guidance scale (set to **3.0**)

### Generation Process

```
1. Start from pure noise: x_T ~ N(0, I)  [shape: (B, 3, 32, 32)]
2. For each timestep t = T, T-1, ..., 1, 0:
   a. Create double batch: [x_t, x_t]
   b. Create double labels: [class_labels, null_labels(10)]
   c. Forward pass → split into ε̂_cond, ε̂_uncond
   d. Apply CFG: ε̂ = ε̂_uncond + w × (ε̂_cond - ε̂_uncond)
   e. Reverse step: x_{t-1} = scheduler.step(ε̂, t, x_t)
3. Post-process: x_0 = (x_0 / 2 + 0.5).clamp(0, 1)
4. Convert to PIL images and display
```

### Generated Samples

The notebook demonstrates generation for **class 3 (Cat)** with a batch size of 4 and guidance scale w = 3.0. Generated images are saved as `output_0.png` through `output_3.png`.

| Parameter | Value |
| --- | --- |
| **Desired Class** | 0 (Plane) |
| **Batch Size** | 4 |
| **Guidance Scale** | 3.0 |
| **Diffusion Steps** | 1000 |
| **Generation Time** | ~21 seconds |

---

## Analysis

### Why Diffusion Models?

Diffusion models have emerged as the state-of-the-art approach for image generation, surpassing GANs in both sample quality and diversity. Key advantages include:

- **Training stability** — No adversarial dynamics; simple MSE loss on noise prediction.
- **Mode coverage** — Diffusion models naturally cover the full data distribution without mode collapse.
- **Controllability** — Class conditioning via embeddings and CFG provides precise control over generated content.

### Classifier-Free Guidance (CFG)

CFG is a powerful technique that improves sample quality by trading off diversity for fidelity:

- **Without CFG (w=1):** The model generates diverse, clear images but failed to adhere to class
- **With CFG (w=3–5):** The model generates class-adherent images by amplifying the conditional signal relative to the unconditional baseline. (but it also kinda just generates blobs sometimes).

- **Too high guidance (w>7):** creates random blobs (idk)

The choice of **w=3.0** balances quality and diversity for CIFAR-10's 32×32 resolution.

### Fine-Tuning Strategy

Rather than training a conditional diffusion model from scratch (which would require significant compute), this approach:

1. **Leverages a pre-trained unconditional model** (`google/ddpm-cifar10-32`) that already understands CIFAR-10's visual statistics.
2. **Patches it minimally** with a class embedding — only the new embedding weights need to be learned from scratch.
4. **Uses CFG dropout** (p=0.1) during training so the model learns both conditional and unconditional generation in a single model.

## Note on LLM Usage

Aside from research and using GitHub Copilot's code autocomplete feature, LLMs were used to help generate code templates for this README and for debugging purposes during development.
