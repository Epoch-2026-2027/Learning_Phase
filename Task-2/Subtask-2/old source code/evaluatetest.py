import torch
import torchvision
from diffusers import DDPMPipeline
from models import device
import tqdm



torch.backends.cudnn.benchmark = True


manual = ''

timesteps = 500
scheduler = 'ddpm' # ddpm or ddim
beta_schedule = 'squaredcos_cap_v2' # ['linear', 'scaled_linear', 'squaredcos_cap_v2', 'sigmoid']

batch_size = 5

pipeline = DDPMPipeline.from_pretrained(f"Saved Models/ddpm-finetuned-{scheduler}-{beta_schedule}-{timesteps}" if not manual else manual).to(device) #


sample = torch.randn(1, 3, 32, 32).to(device)

intermediate_images = []

for t in tqdm.tqdm(pipeline.scheduler.timesteps, total=timesteps):
    with torch.no_grad():
        interm = pipeline.unet(sample, t).sample
    sample = pipeline.scheduler.step(interm, t, sample).prev_sample
    if t%50==0:
        intermediate_images.append(sample.clone().squeeze(dim=0))

intermediates_normalized = [(img + 1) / 2 for img in intermediate_images]

Grid = torchvision.utils.make_grid(intermediates_normalized, padding=25, nrow=3)
img = torchvision.transforms.ToPILImage()(Grid)
img.show()