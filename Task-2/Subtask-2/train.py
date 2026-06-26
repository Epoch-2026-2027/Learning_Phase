import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import io
import PIL
import tqdm
import matplotlib.pyplot as plt
import time
import random

from models import device, ImageSet, DDPM_Model

print("Imported Libraries!")

torch.backends.cudnn.benchmark = True


# Setting a manual seed for reproducability (useful for ablations)
torch.manual_seed(67)

def seed_worker(worker_id): # Manual seeding init function for dataloaders
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)

print("Fixed seed!")


num_epochs = 20
lr = 1e-5
batch_size = 32                 # With gradient accumulation, the effective batch_size = 64
grad_accumulation_steps = 2     #
patience = 6

lr_f = f"{lr:.0e}".replace("e-0","e-").replace("e+0","e")

train_steps = 500
scheduler = 'ddpm' # ddpm or ddim
beta_schedule = 'squaredcos_cap_v2' # ['linear', 'scaled_linear', 's5quaredcos_cap_v2', 'sigmoid']


train_dataloader = DataLoader(ImageSet('train'), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
val_dataloader = DataLoader(ImageSet('val'), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)

print("Loaded dataloaders!")


manual = ""
model = DDPM_Model(train_steps, scheduler, beta_schedule, dropout=None)
model.to(device)


print("Initialised model!")

losses = []
val_losses = []
best_loss=float('inf')
optimizer = torch.optim.AdamW(model.image_pipe.unet.parameters(), lr=lr)
start_time = time.perf_counter()

print("Commencing training/fine-tuning...")
# Fine tuning loop
pat_count=0
for epoch in range(num_epochs):
    val_loss = 0
    train_loss = 0
    i=0
    for step, (X, y) in enumerate(tqdm.tqdm(train_dataloader, total=len(train_dataloader))):
        clean_images = X.to(device)
        
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]


        timesteps = torch.randint(
            0,
            model.image_pipe.scheduler.config.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()


        noisy_images = model.image_pipe.scheduler.add_noise(clean_images, noise, timesteps)
        # [DEBUG] print(next(model.image_pipe.unet.parameters()).device)
        # [DEBUG] print(torch.cuda.is_available())
        # [DEBUG] print(noisy_images.device)
        # [DEBUG] print(timesteps.device)
        noise_pred = model.image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

        loss = F.mse_loss(noise_pred, noise)

        train_loss += loss.item()

        loss.backward()

        if (step + 1) % grad_accumulation_steps == 0: # Gradient accumulation
            optimizer.step()
            optimizer.zero_grad()

        i+=1
    train_loss/=i+1e-8

    i=0
    with torch.no_grad():
        for  X, y in tqdm.tqdm(val_dataloader, total=len(val_dataloader)):
            clean_images = X.to(device)
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                model.image_pipe.scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            noisy_images = model.image_pipe.scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model.image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

            loss = F.mse_loss(noise_pred, noise)  

            val_loss += loss.item()
            i+=1
    val_loss/=i+1e-8

    if val_losses!=[] and val_loss>=best_loss: # Patience mechanism
        if pat_count==patience:
            break
        else:
            pat_count+=1
    else:
        pat_count=0

    if val_loss<best_loss:
        best_loss = val_loss
        model.image_pipe.save_pretrained(f'./Saved Models/{f"ddpm-finetuned-{scheduler}-{beta_schedule}-{train_steps}-{lr_f}-{scheduler}" if not manual else manual}')


    losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Saving functionality

    elapsed_time = time.perf_counter() - start_time

    print(f"(Time Elapsed:{elapsed_time//60:.0f}:{elapsed_time%60:.0f})\tEpoch {epoch+1}, Loss: {train_loss:.4f},  Val Loss: {val_loss:.4f}")

# Plotting
ax = plt.figure().gca()
ax.xaxis.get_major_locator().set_params(integer=True)
plt.plot(losses)
plt.plot(val_losses)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss History')
plt.savefig(f'./Graphs/{f"ddpm-finetuned-{scheduler}-{beta_schedule}-{train_steps}-{lr_f}-{scheduler}" if not manual else manual}_Loss_History.png')