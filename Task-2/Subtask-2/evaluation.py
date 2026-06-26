import torch
import torchvision
import torchvision.transforms as T
from models import device, ImageSet
from diffusers import DDIMScheduler, DDPMScheduler, DDPMPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import tqdm
import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark = True


# Setting a manual seed for reproducability
torch.manual_seed(67)

def seed_worker(worker_id): # Manual seeding init function for dataloaders
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)

print("Fixed seed!")


def generate_samples(pipeline, n_samples, batch_size=32, num_inference_steps=500): 
    pipeline.unet.eval()
    all_images = []
    remaining = n_samples
    with torch.no_grad():
        while remaining>0:
            batch = min(batch_size, remaining)
            output = pipeline(batch_size=batch, num_inference_steps=num_inference_steps, output_type='np')
            img = output.images
            all_images.append(torch.tensor(img, dtype=torch.float))
            remaining-=batch
    return torch.cat(all_images[:n_samples], dim=0).permute((0,3,1,2))

def temperature_generate_samples(pipeline, n_samples, temp, batch_size=32, num_inference_steps=500):
    pipeline.unet.eval()
    all_images = []
    remaining = n_samples
    with torch.no_grad():
        while remaining>0:
            batch = min(batch_size, remaining)
            sample = (torch.randn(batch, 3, 32, 32)*temp).to(device)
            for t in tqdm.tqdm(pipeline.scheduler.timesteps, total=num_inference_steps):
                with torch.no_grad():
                    interm = pipeline.unet(sample, t).sample
                sample = pipeline.scheduler.step(interm, t, sample).prev_sample
            sample = (sample.clamp(-1,1)+1)/2
            all_images.append(sample.cpu())
            remaining-=batch
    return torch.cat(all_images[:n_samples], dim=0).permute((0,3,1,2))


def get_real_samples(n_samples, batch_size=32):
    loader = DataLoader(ImageSet('test'), batch_size=batch_size, shuffle=True)
    all_images = []
    collected = 0
    for X, _ in loader:
        all_images.append(X)
        collected += X.shape[0]
        if collected>=n_samples:
            break
    return torch.cat(all_images[:n_samples], dim=0)


def tensor_to_uint8(imgs):
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    imgs = (imgs * 255).to(torch.uint8)
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    return imgs.to(device)
 
def compute_fid(real_imgs, fake_imgs, feature=2048):
    fid = FrechetInceptionDistance(normalize=False, feature=feature).to(device)
    real_imgs = tensor_to_uint8(real_imgs)
    fake_imgs = tensor_to_uint8(fake_imgs)
    fid.update(real_imgs, real=True)
    fid.update(fake_imgs, real=False)
    return fid.compute().item()

def compute_kid(real_imgs, fake_imgs, feature=2048, subset_size=50): # using KID over FID for this subtask
    kid = KernelInceptionDistance(subset_size=subset_size, normalize=False, feature=feature).to(device)
    real_imgs = tensor_to_uint8(real_imgs)
    fake_imgs = tensor_to_uint8(fake_imgs)
    kid.update(real_imgs, real=True)
    kid.update(fake_imgs, real=False)
    return kid.compute()

def make_sample_grid(pipeline, path, inf_steps, n_row, temp=1, title=None):
    sample = torch.randn(1, 3, 32, 32).to(device) * temp

    intermediate_images = []

    for i,t in enumerate(tqdm.tqdm(pipeline.scheduler.timesteps, total=inf_steps)):
        with torch.no_grad():
            interm = pipeline.unet(sample, t).sample
        sample = pipeline.scheduler.step(interm, t, sample).prev_sample
        if i%(inf_steps//5)==0:
            intermediate_images.append(sample.clone().squeeze(dim=0))

    intermediates_normalized = [(img + 1) / 2 for img in intermediate_images]

    grid = torchvision.utils.make_grid(intermediates_normalized, padding=25, nrow=n_row)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved sample grid to {path}")



def evaluate(pipeline, n_real=1000, n_gen=1000, num_inference_steps=1000,
             grid_path='./Graphs/sample_grid.png', label=None, n_row=3, temp=None): #
    print(f"Generating {n_gen} samples...")
    pipeline.scheduler.set_timesteps(num_inference_steps)
    if temp:
        gen_imgs = temperature_generate_samples(pipeline, n_gen, temp, num_inference_steps=num_inference_steps)
    else:
        gen_imgs = generate_samples(pipeline, n_gen, num_inference_steps=num_inference_steps)
        temp=1
    # [DEBUG] print(gen_imgs.shape)
 
    print(f"Loading {n_real} real samples...")
    real_imgs = get_real_samples(n_real)
    # [DEBUG] print(real_imgs.shape)
 
    # print("Computing FID...")
    # fid_score = compute_fid(real_imgs, gen_imgs)
    # print(f"FID: {fid_score:.4f}")

    kid_score, kid_std = compute_kid(real_imgs, gen_imgs, subset_size=50)
    print(f"KID SCORE: {kid_score:.4f}, KID STD: {kid_std:.4f}")
 
    title = label if label else f"Generated Samples (steps={num_inference_steps})"
    make_sample_grid(pipeline, grid_path, num_inference_steps, n_row, temp, title)
 
    return kid_score, kid_std, gen_imgs

if __name__=='__main__':
    torch.backends.cudnn.benchmark = True
    manual = ''

    lr = 1e-5
    timesteps = 500 # this refers to the trained model's timesteps
    scheduler = 'ddpm' # ddpm or ddim
    beta_schedule = 'squaredcos_cap_v2' # ['linear', 'scaled_linear', 'squaredcos_cap_v2', 'sigmoid']
    temp = 100

    lr_f = f"{lr:.0e}".replace("e-0","e-").replace("e+0","e")

    pipeline = DDPMPipeline.from_pretrained("Saved Models/"+(f"ddpm-finetuned-{scheduler}-{beta_schedule}-{timesteps}-{lr_f}-{scheduler}" if not manual else manual)).to(device) #

    path = "Generations/" + (f"ddpm-finetuned-{scheduler}-{beta_schedule}-{timesteps}-{lr_f}-{scheduler}" if not manual else manual) + "/"
    inference_steps = 100

    if not os.path.exists(path):
        os.makedirs(path)

    fid, std, samples = evaluate(
        pipeline,
        n_real=128,
        n_gen=128,
        num_inference_steps=inference_steps,
        grid_path= path+f'samplegrid_{inference_steps}{'' if not temp else f'_temp{temp}'}.png',
        label=f'Fine-tuned DDPM ({inference_steps} steps)'
    )
    torchvision.utils.save_image(samples, path+f'generatedimages_{inference_steps}{'' if not temp else f'_temp{temp}'}.png')