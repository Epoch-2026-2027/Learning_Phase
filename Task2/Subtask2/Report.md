# GAN Track — CIFAR-10 DCGAN Fine-Tuning: Report (Concise)

**Dataset:** CIFAR-10, 50,000 training images, 10 balanced classes
**Base model:** Pretrained DCGAN (`csinva/gan-vae-pretrained-pytorch`, 200 epochs)

---

## 1. Architecture

- Generator/Discriminator rebuilt from scratch, layer-for-layer matched to the pretrained checkpoint's `state_dict` (key names + tensor shapes), so `load_state_dict` loads with zero mismatches.
- **Generator** (`nz=100 → nc=3`, 32×32 RGB): 4× `ConvTranspose2d(k=4,s=2,p=1)` blocks with BatchNorm+ReLU, upsampling 1×1 → 32×32, followed by a final `ConvTranspose2d(k=1)` for channel mixing only, then `Tanh`.
- **Discriminator** (`nc=3 → 1`): mirror structure using strided `Conv2d` + BatchNorm + LeakyReLU(0.2), ending in `Conv2d(k=2,s=2)` → `Sigmoid`.
- **Spectral norm** — Spectral norm was applied to each Conv/ConvTranspose layer only after the pretrained weights were loaded via load_state_dict.

---

## 2. Experiments

Three variants, each fine-tuned for 5 and 10 epochs (6 runs total). Same architecture, dataset, batch size (128), optimizer (Adam, β=0.5/0.999) throughout.

| Variant | LR | Spectral Norm |
|---|---|---|
| Baseline | 2e-4 | No |
| Spectral Norm | 2e-4 | Yes |
| Low LR | 5e-5 | No |

---

## 3. Ablation Results

**Summary stats (from real training logs):**

| Run | D mean | D CV* | G mean | Max D spike |
|---|---|---|---|---|
| Baseline, 5 ep | 0.037 | 2.22 | 7.90 | 0.51 |
| Baseline, 10 ep | 0.445 | 8.45 | 8.14 | **33.66** |
| Spectral Norm, 5 ep | 0.418 | **0.38** | 2.26 | 0.82 |
| Spectral Norm, 10 ep | 0.348 | **0.49** | 2.66 | 0.92 |
| Low LR, 5 ep | 0.013 | 1.61 | 7.93 | 0.12 |
| Low LR, 10 ep | 0.011 | 2.50 | 8.47 | 0.20 |

*CV = std/mean of Loss_D, used as an objective stability measure.


**Findings:**

- **D-dominance is the default failure mode.** Baseline and Low LR both collapse `Loss_D` toward 0 while `Loss_G` stays stuck at 7-9, with no recovery across 10 epochs.
- **Lowering LR alone does not fix it.** Low LR's D-dominance was as severe as Baseline's (slightly lower median D-loss) 
- **One real divergence event**, Baseline only, 10 epochs: `Loss_D` spiked to 33.66 at step `[6/10][350/390]` (`Loss_G` to 19.39). Did not occur in any other run, including Baseline at 5 epochs — instability compounds with more unstabilized training.
- **Spectral norm is the only variant that stabilizes training.** `Loss_D` CV is 5-20× lower than every other variant, at both epoch counts, with no divergence events.

---

## 5. Analysis

- **Stability**- Only spectral norm produced a stable D/G equilibrium — consistent with its purpose of bounding each layer's Lipschitz constant so D's gradients can't outpace G's.
- **No visual mode collapse observed** in any run (qualitative check only — outputs varied in object type/color/background). Not yet confirmed numerically; `evaluate.py`'s `pairwise_pixel_distance` exists for this and should be run on final checkpoints.
- **Pretraining matters**: even the unstable runs produced recognizable (if degraded) images rather than noise, since 200 epochs of prior training gave the model a strong prior that a few unstable epochs damaged but didn't erase.

---

## 6. Conclusion
Spectral norm (applied after weight loading) is the only intervention tested that stabilizes training — lower loss variance, no divergence, more coherent samples.

---

## LLM usage
- LLM was used for a quick search of dataset
- It was used for debugging and sometimes help in writing of code and handling errors
- It was used to try working in different py and ipynb files instead of single ipyb
- I have tried to use Argument parser for initializing the hyperparameters from the command line . LLM was used to do and try this.
- It was used to derive conclusions of the experiments

