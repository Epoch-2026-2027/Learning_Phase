# Subtask 1 — Dilated U-Net for Satellite Road Extraction

## Overview


It explores semantic segmentation by implementing a **U-Net with a Dilated Cascade Bottleneck** to predict pixel-wise masks for road extraction from satellite imagery.  This implementation uses the [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset). The entire pipeline for the custom U-net — data loading, model definition, training, and evaluation — is built in **JAX/Flax (NNX)** with the **Grain** data loading framework.
However, the final model with pre-trained weights uses tensorflow instead, as it provides much better support for fine-tuning.

---

## Repository Structure

```text
Task2/Subtask1/
├── models/                        # Directory for saved model checkpoints (Orbax)
├── sample_images/                 # Sample satellite images for quick inference demos
├── task2-subtask1.ipynb           # JAX/Flax implementation: data pipeline, baseline U-Net, and D-LinkNet models
├── task2-subtask1-tf.ipynb        # TensorFlow implementation: Final model with pre-trained ResNet50 backbone
└── README.md                     # This file
```

---

## Usage Instructions

### Implementations & Framework Switch
This subtask includes two separate implementations that are both necessary for the task:

1. **`task2-subtask1.ipynb` (JAX/Flax NNX):** Contains the original baseline U-Net and D-LinkNet models.
2. **`task2-subtask1-tf.ipynb` (TensorFlow):** Contains the final model using a pre-trained ResNet50 backbone.

**Why the framework switch?**
The initial pipeline was built in JAX using Flax NNX. However, there were no pre-trained image models available natively in the Flax NNX API, which made it extremely difficult to transpile a pre-trained model for use as a backbone. To successfully implement the pre-trained backbone architectural improvement required by the assignment, the final model was built in TensorFlow/Keras where pre-trained ResNet50 weights are readily available.

### Training & Evaluation (`task2-subtask1.ipynb` Pipeline)
Execute the notebook cells sequentially to:

1. **Build the data pipeline** using Grain's `RandomAccessDataSource` with deterministic validation anchoring for reproducible crops.
2. **Define the model** — a standard U-Net enhanced with a Dilated Cascade Bottleneck at the deepest encoder level.
3. **Train for 60 epochs** using BCE + Dice loss with a cosine decay learning rate schedule (AdamW, initial lr = 1e-3, weight decay = 1e-4).
4. **Evaluate** on the held-out validation set using global Precision, Recall, F1 (Dice), and IoU metrics.
5. **Visualize predictions** overlaid on original satellite images.

---

## Dataset

- **Source:** DeepGlobe Road Extraction Dataset (Kaggle)
- **Images:** 6,226 satellite images at 1024×1024 resolution
- **Split:** 4,980 training / 1,246 validation (80/20 split via `train_test_split`)
- **Patch Size:** 256×256 random crops during training; deterministic targeted crops during validation (anchored to road pixel locations for reliable metric computation)
- **Masks:** Binary masks (`_mask.png`) with road pixels > 0

---

## Model Architecture

### Dilated U-Net (`DilatedUNet`)

The architecture is a classic encoder-decoder U-Net with skip connections, modified with a **Dilated Cascade CNN** at the bottleneck to dramatically expand the receptive field without additional pooling.

```
Input (B, 256, 256, 3)
       │
   ┌───▼───┐
   │ enc1  │  DoubleConv(3 → 64)        ──────────────────────┐
   └───┬───┘                                                   │
   MaxPool(2×2)                                                │
   ┌───▼───┐                                                   │
   │ enc2  │  DoubleConv(64 → 128)       ─────────────────┐   │
   └───┬───┘                                               │   │
   MaxPool(2×2)                                            │   │
   ┌───▼───┐                                               │   │
   │ enc3  │  DoubleConv(128 → 256)      ────────────┐    │   │
   └───┬───┘                                          │    │   │
   MaxPool(2×2)                                       │    │   │
   ┌───▼───┐                                          │    │   │
   │ enc4  │  DoubleConv(256 → 512)      ───────┐    │    │   │
   └───┬───┘                                     │    │    │   │
   MaxPool(2×2)                                  │    │    │   │
   ┌───▼───────────┐                             │    │    │   │
   │  Bottleneck   │  DilatedCNN(512)            │    │    │   │
   │  d1(r=1) →    │  Cascaded dilated convs     │    │    │   │
   │  d2(r=2) →    │  with residual aggregation  │    │    │   │
   │  d4(r=4) →    │  x + out1 + out2 + out3     │    │    │   │
   │  d8(r=8)      │  + out4                     │    │    │   │
   └───┬───────────┘                             │    │    │   │
   ConvTranspose(2×2)                            │    │    │   │
   ┌───▼───┐                                     │    │    │   │
   │ dec4  │  Concat(↑, enc4) → DoubleConv ◄─────┘    │    │   │
   └───┬───┘                                          │    │   │
   ConvTranspose(2×2)                                 │    │   │
   ┌───▼───┐                                          │    │   │
   │ dec3  │  Concat(↑, enc3) → DoubleConv ◄──────────┘    │   │
   └───┬───┘                                               │   │
   ConvTranspose(2×2)                                      │   │
   ┌───▼───┐                                               │   │
   │ dec2  │  Concat(↑, enc2) → DoubleConv ◄───────────────┘   │
   └───┬───┘                                                   │
   ConvTranspose(2×2)                                          │
   ┌───▼───┐                                                   │
   │ dec1  │  Concat(↑, enc1) → DoubleConv ◄───────────────────┘
   └───┬───┘
   ┌───▼───────┐
   │ 1×1 Conv  │  Conv(64 → 1)  →  Logits
   └───────────┘
       │
   Sigmoid → Binary Mask
```

### Key Components

| Component | Description |
| --- | --- |
| **DoubleConv** | Two sequential `Conv(3×3, SAME) → BatchNorm → ReLU` blocks |
| **Down (Encoder)** | `MaxPool(2×2) → DoubleConv` — halves spatial dimensions |
| **Up (Decoder)** | `Bilinear Resize(2×) → Conv(3×3) → BN → ReLU → Concat(skip) → DoubleConv` |
| **DilatedCNN (Bottleneck)** | Cascade of 4 dilated convolutions (r=1,2,4,8) with residual aggregation: `x + Σ outputs` |
| **Output Head** | `Conv(1×1)` mapping 64 channels → 1 (binary logit) |

### Why D-LinkNet (Dilated Convolutions at the Bottleneck)?

The D-LinkNet architecture was specifically chosen to address the unique challenges of road extraction. Standard U-Nets use a basic max-pool + double-conv block at the bottleneck, which limits the effective receptive field. Roads are continuous, long, and can span across the entire image. By replacing the bottleneck with **cascaded atrous convolutions** (dilation rates 1, 2, 4, 8) as proposed in D-LinkNet, the network:

1. **Expands the receptive field exponentially** without adding pooling layers (which would destroy crucial spatial resolution).
2. **Captures multi-scale context and long-range dependencies** — essential for maintaining the connectivity of roads (varying from narrow alleys to multi-lane highways) even when partially occluded by trees or shadows.
3. **Avoids the "gridding effect"** through sequential (rather than parallel) cascading of increasing dilation rates.
4. **Preserves high-frequency local texture** via residual aggregation (`x + out1 + out2 + out3 + out4`).

---

## Data Pipeline & Cropping Strategies

A significant difference between the implementations lies in their data cropping strategies for evaluation.

### JAX Pipeline (Grain) — Optimal Cropping
The JAX pipeline (`task2-subtask1.ipynb`) uses **Grain** (`grain.python`) for deterministic data loading, focusing on maximizing useful signal:
- **`RandomCrop` (Training):** Extracts 256×256 patches randomly.
- **Validation Anchoring (Optimal Crop):** Random cropping during validation can result in patches with 0% road pixels, introducing massive variance into metrics like IoU. To solve this, `generate_validation_anchors()` was used to pre-compute deterministic crop coordinates centered *only* on road pixels. This **optimal crop** ensures that every validation patch contains roads, enabling highly reliable and stable metric computation.

### TensorFlow Pipeline — CenterCrop
In contrast, the TensorFlow pipeline (`task2-subtask1-tf.ipynb`) utilizes standard `tf.data` utilities with a **CenterCrop** strategy for evaluation. While simpler to implement, it risks evaluating patches that consist entirely of background (e.g., dense forest or water). This can artificially inflate overall pixel accuracy (due to true negatives) while severely punishing intersection-based metrics (Precision/IoU) if the model hallucinates even a single road pixel in an otherwise empty patch. 
But, training ended up stable at crop size 512, it's very unlikely that there are no roads in the center 1/4th of the image. 
Plus, the reason I did this was to align the validation data more closely with the training random crop.

---

## Loss Function

The model is trained with a **combined BCE + Dice Loss**:

```
L = BCE(logits, labels) + DiceLoss(sigmoid(logits), labels)
```

- **Binary Cross-Entropy (BCE):** Provides pixel-wise gradient signal, especially valuable for the dominant background class.
- **Dice Loss:** Directly optimizes the overlap metric (F1/Dice score), counteracting the severe class imbalance in road segmentation (roads typically occupy < 5% of pixels).

- **Why:-** Using only BCE loss in such a huge class imbalance will likely result in the model just predicting the background class, (model collapse), while only DICE may lead to unstable gradients.

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| **Framework** | JAX / Flax NNX |
| **Optimizer** | AdamW (weight decay = 1e-4) |
| **Learning Rate Schedule** | Cosine decay (init = 1e-3, decay steps = 10,000, alpha = 0.01) |
| **Epochs** | 60 |
| **Batch Size** | 4 |
| **Patch Size** | 256 × 256 |
| **Training Set** | 4,980 images |
| **Validation Set** | 1,246 images |
| **Data Workers** | 6 (Grain) |
| **Loss** | BCE + Dice |
| **Accelerator** | GPU (Kaggle T4) |

---

## Training Results

### Loss Curve (60 Epochs)

| Epoch | Train Loss | Val Loss | Time (s) |
| --- | --- | --- | --- |
| 1 | 1.0966 | 1.1783 | 201.6 |
| 5 | 0.8154 | 0.7383 | 150.1 |
| 10 | 0.7130 | 0.5842 | 149.6 |
| 15 | 0.6529 | 0.5506 | 148.2 |
| 18 | 0.6224 | **0.5318** | 148.7 |
| 20 | 0.5975 | 0.5495 | 147.1 |
| 30 | 0.5352 | 0.5839 | 149.2 |
| 40 | 0.5267 | 0.5856 | 150.4 |
| 50 | 0.5213 | 0.5924 | 148.6 |
| 60 | 0.5165 | 0.5996 | 150.1 |

**Key Observations:**
- **Best validation loss** was achieved around **Epoch 18** (val loss = 0.5318), after which the validation loss plateaus and gradually increases — a classic sign of mild overfitting.
- Training loss continues to decrease monotonically, reaching 0.5165 at epoch 60.
- Each epoch takes approximately **148–150 seconds** on a Kaggle T4 GPU (with the first epoch slower due to JIT compilation).
- The model exhibits strong convergence in the first 20 epochs, with diminishing returns thereafter.

### Evaluation Metrics

The model is evaluated on the validation set using global confusion-matrix-based metrics (threshold = 0.5):

| Metric | Description |
| --- | --- |
| **Precision** | TP / (TP + FP) — How many predicted road pixels are actually roads |
| **Recall** | TP / (TP + FN) — How many actual road pixels were detected |
| **F1 (Dice)** | 2 × Precision × Recall / (Precision + Recall) — Harmonic mean |
| **IoU (Jaccard)** | TP / (TP + FP + FN) — Intersection over Union |

### Final Evaluation Scores

| Model | Framework | Precision | Recall | F1 (Dice) | IoU |
| --- | --- | --- | --- | --- | --- |
| **Baseline U-Net** | JAX | 0.7954 | 0.6526 | 0.7170 | 0.5588 |
| **D-LinkNet** | JAX | 0.6409 | 0.6736 | 0.6569 | 0.4891 |
| **ResNet50 U-Net** | TensorFlow | 0.7567 | 0.7496 | 0.7531 | 0.6040 |

#### **Note**: an analysis of the scores may lead you to question why D-linknet had lesser score. the problem was with the encoder layers that still use max pooling to save training time, which hinders the main advantage of maintaining global context that the dilated convolutions in the bottleneck provide. Also the increased parameters showed sign of overfitting.

## Visualization

The notebook includes a `visualize_prediction()` function that generates side-by-side comparison plots:

1. **Input Satellite Image** — The original RGB satellite tile
2. **Ground Truth Mask** — The binary road mask (white = road)
3. **Predicted Mask** — The model's thresholded output (threshold = 0.5)
4. **Prediction Overlay** — Red highlights blended onto the original image (alpha = 0.6)

Predictions are saved to `/kaggle/working/visualisations/`.

---

## Experimentation & Design Decisions

As per the assignment requirements for **Architectural Improvements** and **Training and Loss Function Experiments**, the following design decisions were made:

- **Architectural Improvements (Pre-trained Backbones & D-LinkNet):** To fulfill the architectural improvements requirement, two approaches were taken. First, a D-LinkNet (U-Net with a dilated cascade bottleneck) was implemented in JAX/Flax to expand the receptive field without losing spatial resolution. Second, to leverage transfer learning, a **pre-trained ResNet50 backbone** was implemented in TensorFlow (`task2-subtask1-tf.ipynb`).

- **Loss Function Experiment (BCE + Dice):** A combined BCE + Dice loss was selected. Binary Cross-Entropy (BCE) provides stable pixel-wise gradient signals for the dominant background class, while Dice Loss directly optimizes the overlap metric, counteracting the severe class imbalance inherent in road segmentation.

- **Deterministic Validation Anchoring:** Random cropping during validation would introduce evaluation noise. The `generate_validation_anchors()` function pre-computes deterministic crop coordinates centered on road pixels, ensuring consistent and meaningful metric computation across training runs.

- **Grain Data Pipeline:** Grain was used instead of `tf.data` or PyTorch `DataLoader` because it provides native JAX compatibility with deterministic, reproducible data loading — critical for scientific reproducibility.

- **Cosine Decay Schedule:** The cosine decay schedule was preferred over step-based decay for its smooth learning rate reduction, which typically leads to better generalization in segmentation tasks.

- **D4 TTA:** I tried averaging the predictions across D4 group actions in the evaluation phase, which is a form of test time augmentation (TTA). This helps to improve the model's performance by averaging the predictions across different transformations of the same input. Note that taking direct average actually hurt my score because the model regularly generates low confidence masks. taking the maximum or reducing the threshold may lead to better results. the process increased inference time 8 fold, so I couldn't get the oppurtunity to properly test optimal threshold values.

---

## Note on LLM Usage

Aside from research and using GitHub Copilot's code autocomplete feature, LLMs were used to help generate code templates for this README and for debugging purposes during development. Also code comments are AI-generated. (except some of them)

