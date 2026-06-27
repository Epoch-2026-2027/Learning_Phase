# Usage Commands — GAN Track (CIFAR-10 DCGAN)

| Command | When to use |
|---|---|
| `python model.py` | Verify the Generator/Discriminator load the pretrained checkpoint correctly |
| `python train.py --epochs 1 --outf ./test_output` | Quick 1-epoch pipeline check, no spectral norm |
| `python train.py --epochs 1 --spectral_norm --outf ./test_output_sn` | Quick 1-epoch pipeline check, with spectral norm |
| `python train.py --epochs 5 --lr 2e-4 --outf ./runs/baseline_e5` | run: baseline, 5 epochs |
| `python train.py --epochs 10 --lr 2e-4 --outf ./runs/baseline_e10` | run: baseline, 10 epochs |
| `python train.py --epochs 5 --lr 2e-4 --spectral_norm --outf ./runs/sn_e5` |run: spectral norm, 5 epochs |
| `python train.py --epochs 10 --lr 2e-4 --spectral_norm --outf ./runs/sn_e10` | run: spectral norm, 10 epochs |
| `python train.py --epochs 5 --lr 5e-5 --outf ./runs/lowlr_e5` | run: low LR, 5 epochs |
| `python train.py --epochs 10 --lr 5e-5 --outf ./runs/lowlr_e10` | run: low LR, 10 epochs |
| Add `2>&1 \| tee <path>/log.txt` to any `train.py` command | Save the printed loss log to a file instead of just the terminal |

