# Generative Modeling on CIFAR-10 — Diffusion Models

**Epoch — The AI/ML Club, IIT Hyderabad**

**Track:** Diffusion Models

**Author:** Swayam Verma

---

## Assignment

This submission fine-tunes and analyzes a Denoising Diffusion Probabilistic Model (DDPM) on CIFAR-10, as assigned to the Diffusion Models track. Per the assignment's model-initialization requirement, a publicly available pretrained checkpoint (`google/ddpm-cifar10-32`) is wrapped inside a custom PyTorch codebase rather than used through a ready-made generation pipeline:

1. **Pretrained weights** — loaded from `google/ddpm-cifar10-32` on the Hugging Face Hub.
2. **Re-implementation/wrapping** — the U-Net backbone is wrapped in a custom `DiffusionModel` class; the noise scheduler (forward diffusion, DDPM ancestral sampling, DDIM sampling) is implemented from scratch in `schedulers.py` rather than using `diffusers`' built-in scheduler classes.
3. **Loading pretrained weights** — into the wrapped architecture.
4. **Fine-tuning** — on CIFAR-10, with two ablations layered on top (noise schedule, inference speed) plus a class-specific fine-tuning experiment.
5. **Experiments and analysis** — covered in [REPORT.md](REPORT.md).

> 📄 Full experimental report (architecture, ablations, analysis): **[REPORT.md](REPORT.md)**

![Speed vs. quality comparison](assets/speed_quality_comparison.png)

## Suggested-workflow checklist

| Brief section | Where it's covered |
|---|---|
| Data Exploration (class distribution, image characteristics, intra-class variation, preprocessing) | REPORT.md §2 |
| Baseline Generation (pretrained DDPM, initial samples) | REPORT.md §3 |
| Model Improvements — noise schedules | REPORT.md §4 (Ablation 1: linear vs. cosine) |
| Model Improvements — inference speed | REPORT.md §5 (Ablation 2: DDPM vs. DDIM) |
| Analysis (quality, diversity, training stability, sampling quality) | REPORT.md §4–7 |
| Ablations — class-specific training | REPORT.md §6 |
| Evaluation methodology | REPORT.md §7 |

## Repo structure

```
.
├── Generative_Modelling.ipynb   # End-to-end notebook: data exploration → baseline → ablations
├── src/
│   ├── model.py                 # DiffusionModel wrapper around the pretrained U-Net
│   ├── schedulers.py            # Forward process + DDPM/DDIM sampling, implemented from scratch
│   ├── train.py                 # Fine-tuning loop (CLI)
│   ├── sample.py                # Generation script, DDPM or DDIM (CLI)
│   ├── data_utils.py            # CIFAR-10 loading, class distribution, sample grids
│   └── evaluate.py               # Diversity score, Inception-distance proxy, loss-curve comparison
├── assets/                      # Figures referenced in this README and the report
├── requirements.txt
└── REPORT.md                    # Experimental report
```

## Setup

```bash
pip install -r requirements.txt
```

Tested on a Colab T4 GPU. The notebook expects all files in `src/` to be on the Python path alongside it (Colab convention: upload the contents of `src/` flat into `/content/`, or run from inside `src/`), since the modules import each other directly (e.g. `from model import DiffusionModel`).

## Usage

**Fine-tune:**
```bash
python src/train.py --schedule cosine --epochs 5 --batch_size 64 --lr 1e-5 --output_dir outputs
python src/train.py --schedule cosine --classes automobile --epochs 30 --lr 5e-5 --output_dir outputs
```

**Generate:**
```bash
# Fast sampling (DDIM, 50 steps)
python src/sample.py --checkpoint outputs/unet_cosine.pt --method ddim --steps 50 --num_images 16

# Full-fidelity reference sampling (DDPM, 1000 steps)
python src/sample.py --checkpoint outputs/unet_cosine.pt --method ddpm --num_images 16
```

**Compare ablation runs:**
```python
from evaluate import compare_loss_curves
compare_loss_curves(history_glob="outputs/loss_*.json", save_path="outputs/loss_comparison.png")
```

Or open [`Generative_Modelling.ipynb`](Generative_Modelling.ipynb) for the full guided walkthrough (data exploration → baseline → both ablations → class-specific fine-tuning), as run on Colab — this is the "easy to use demonstration" deliverable.

## Key results (summary)

| Experiment | Finding |
|---|---|
| Linear vs. cosine schedule | Linear schedule converged to a noticeably lower, more stable training loss under identical fine-tuning settings; cosine showed higher loss and visibly noisier samples in this setup |
| DDPM (1000 steps) vs. DDIM (10/50/200 steps) | DDIM at 50 steps captured nearly all of the perceptual quality of full 1000-step DDPM at ~1/20th the wall-clock cost; quality degraded sharply by 10 steps |
| Class-restricted fine-tuning (automobile only) | Narrowing the training set to a single class produced visibly more coherent, higher-fidelity samples for that class than the multi-class baseline |

Full discussion, figures, and the reasoning behind each result are in [REPORT.md](REPORT.md).

## Implementation notes

- **`pred_x0` clipping:** both the DDPM and DDIM reverse steps clip the predicted `x0` to `[-1, 1]` at every step. Without this, schedules with `beta_t` close to 1 near `t = T` (the cosine schedule, in particular) cause the `1/sqrt(alpha_t)` term to amplify small noise-prediction errors every step, and the sampling trajectory diverges well before reaching the final output clamp.
- **Evaluation metrics:** full FID requires a 2048-dim Inception feature pool computed over ~50k samples and reference statistics — not practical for a handful of fine-tuning runs on a single GPU within the assignment's compute budget. Instead this repo uses two lighter, clearly-labeled proxies (`evaluate.py`): a pairwise pixel-distance **diversity score**, and an **Inception-distance** proxy (mean-feature-distance in Inception-v3 embedding space, computed on small sample counts). Both are useful for *relative* comparison between this repo's own runs, not as absolute, paper-comparable numbers.

## Limitations

- Ablations were run with a small number of fine-tuning epochs and modest sample counts per evaluation (compute-constrained, single Colab T4 GPU), so the quality/diversity metrics are directional signals rather than statistically robust estimates.
- The "Inception distance" reported here is explicitly *not* FID — see REPORT.md §7.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Nichol, A., & Dhariwal, P. (2021). [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (cosine schedule).
- Song, J., Meng, C., & Ermon, S. (2020). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM).
- Pretrained checkpoint: [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) on the Hugging Face Hub.
