import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from model import Generator

# LLM was used for functions of evaluate.py
def load_generator(checkpoint_path, device, nz=100, ngf=64, nc=3):
    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
    netG.eval()
    return netG


def generate_samples(netG, noise, device):
    noise = noise.to(device)
    with torch.no_grad():
        samples = netG(noise).cpu()
    return samples


def pairwise_pixel_distance(samples):
    n = samples.size(0)
    flat = samples.view(n, -1)                # (N, 3*32*32)
    dists = torch.cdist(flat, flat, p=2)        # (N, N) pairwise L2 distances
    mask = ~torch.eye(n, dtype=torch.bool)      # exclude diagonal (self-distance = 0)
    return dists[mask].mean().item()


def compare_before_after(pretrained_path, finetuned_path, device, n_samples=64, nz=100):
    noise = torch.randn(n_samples, nz, 1, 1)  # SAME noise for both models -> fair comparison

    netG_pre = load_generator(pretrained_path, device, nz=nz)
    netG_fine = load_generator(finetuned_path, device, nz=nz)

    samples_pre = generate_samples(netG_pre, noise, device)
    samples_fine = generate_samples(netG_fine, noise, device)

    dist_pre = pairwise_pixel_distance(samples_pre)
    dist_fine = pairwise_pixel_distance(samples_fine)

    print(f"Avg pairwise pixel distance — pretrained baseline: {dist_pre:.4f}")
    print(f"Avg pairwise pixel distance — fine-tuned model:    {dist_fine:.4f}")
    if dist_fine < dist_pre * 0.7:
        print("WARNING: fine-tuned model's diversity dropped substantially "
              "relative to baseline -- possible mode collapse.")

    # Side-by-side grid: top half pretrained, bottom half fine-tuned
    grid_pre = vutils.make_grid(samples_pre[:32], nrow=8, normalize=True)
    grid_fine = vutils.make_grid(samples_fine[:32], nrow=8, normalize=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].imshow(grid_pre.permute(1, 2, 0))
    axes[0].set_title("Pretrained baseline")
    axes[0].axis("off")
    axes[1].imshow(grid_fine.permute(1, 2, 0))
    axes[1].set_title("Fine-tuned")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("before_after_comparison.png")
    plt.show()

    return dist_pre, dist_fine


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_before_after(
        pretrained_path="./pretrained/netG_epoch_199.pth",
        finetuned_path="./output/netG_epoch_009.pth",
        device=device,
    )
