# Road Segmentation from Satellite Images — Report

## 1. Dataset & Preprocessing

- **Dataset:** Massachusetts Roads Dataset (TIFF aerial images, 1500×1500, with binary road masks)
- **Class imbalance:** road pixels ≈ 5% of total pixels — heavy imbalance, confirmed via pixel count across mask files
- **Preprocessing pipeline:**
  - Pad 1500×1500 → 1536×1536 (18px border, zero-padding) so it divides evenly into tiles
  - Tile into 256×256 patches (36 patches/image) — chosen over resizing to preserve thin road detail
  - Normalize images to [0,1]; binarize masks to {0,1}
  - Lazy-loading `Dataset` class with image-level caching (avoids reprocessing all 36 patches per `__getitem__` call; cuts epoch time ~20x)


## 2. Baseline Architecture — Custom U-Net

- Built from scratch: originally 5-level, reduced to **3-level** after hitting GPU memory limits (3.65GB VRAM) at 512×512 input
- Switched input patch size 512→256 to fit memory
- Added **BatchNorm2d** after every Conv2d (before ReLU) — original 5-level version without BatchNorm suffered from vanishing gradients / dead activations (loss flatlined at ln(2) ≈ 0.693, indicating the model predicted ~0.5 probability everywhere and never learned)


## 3. Architectural Improvements Tried

| Architecture | Source | Why chosen |
|---|---|---|
| **ResNet34-backbone U-Net** | `segmentation_models_pytorch` (pretrained, ImageNet) | Pretrained features transfer well even with very limited fine-tuning data  |
| **DeepLabV3 (ResNet50)** | `torchvision.models.segmentation` (pretrained) | Atrous convolutions capture multi-scale context — relevant for satellite imagery where roads need both fine local detail and broader spatial context |


## 4. Loss Functions Compared

- **BCE** 
- **Dice Loss** 
- **BCE + Dice** 

Evaluation metric: **Dice score** 

## 5. Experimental Results

All quick-pass runs: `batch_size=8`, `num_epochs=10`

### Scratch U-Net, amount=30 images (~1080 patches)

| Loss | Val Dice (range across epochs) | Behavior |
|---|---|---|
| BCE | 0.05 – 0.08 | Stable but weak; barely detects roads |
| Dice | 0.01 – 0.08, trending down | Severe overfitting — train loss ↓, val Dice collapses |
| BCE+Dice | 0.005 – 0.06, trending down | Same overfitting pattern as Dice alone |

### Scratch U-Net, amount=100 images (~3600 patches), BCE+Dice

| Setting | Val Dice (range) | Behavior |
|---|---|---|
| No weight decay | 0.006 – 0.047 | Overfitting persists — more data alone did not fix it |
| weight_decay=1e-4 | 0.001 – 0.061 | Still overfits — regularization at this strength insufficient |

### Pretrained backbones, amount=30 images, BCE+Dice

| Model | Val Dice (range) | Behavior |
|---|---|---|
| **ResNet34-UNet** | 0.39 – 0.58 | Best and most stable result overall; mild fluctuation, no collapse |
| **DeepLabV3** | 0.26 – 0.58 | Strong but noisier epoch-to-epoch than ResNet-UNet |

## 6. Key Findings

1. **Class imbalance dominates plain BCE performance** — BCE loss decreases steadily even when the model barely detects roads, because correctly predicting the majority class (background) is enough to lower average pixel-wise loss. This validates using Dice or combined losses for this task in principle.
2. **From-scratch U-Net severely overfits at this data scale**, regardless of loss function used (BCE, Dice, or BCE+Dice all show the pattern at amount=30; Dice/BCE+Dice show it more severely). Increasing data 30→100 images and adding weight decay (1e-4) did not resolve it within 10 epochs — likely needs substantially more data, stronger regularization, dropout, or augmentation (untested here).
3. **Pretrained backbones vastly outperform the scratch U-Net** at identical, very limited data (amount=30): ResNet34-UNet and DeepLabV3 reach Val Dice ~0.4–0.6 versus the scratch U-Net's ~0.01–0.08. This is the strongest and most reliable signal across all experiments — pretrained ImageNet features generalize to satellite road segmentation even with minimal fine-tuning data.
4. **ResNet34-UNet was the most stable architecture** tested — less epoch-to-epoch fluctuation than DeepLabV3, likely because its decoder is simpler (standard U-Net upsampling vs. DeepLabV3's atrous-convolution head), reducing capacity for overfitting noise on small data.

## 7. Conclusion 

- For this dataset size and GPU constraint (3.65GB VRAM), **a pretrained-backbone architecture (ResNet34-UNet) is clearly the better choice** over a from-scratch U-Net — it achieves meaningfully higher Dice scores with the same limited data and training time.
- **BCE+Dice** is the most defensible loss choice for the pretrained models, balancing pixel-wise confidence (BCE) with explicit overlap optimization (Dice) for the imbalanced class distribution.
- The from-scratch U-Net's poor performance is attributable to **insufficient training data relative to model capacity**, not faulty architecture design — with significantly more training images (likely the full ~1100-image dataset) and/or stronger regularization (dropout, augmentation), it would be expected to close much of the gap.

## 8. LLM usage
- LLM was majorly used to debug code and solve errors (especially the OOM error)
- It was used to analyse the logs of the models for conclusions 
- Was used to try making the model more better (although it is still overfitting)


