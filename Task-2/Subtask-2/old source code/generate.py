import torch
import torchvision
from diffusers import DDPMPipeline
from models import device
import os



torch.backends.cudnn.benchmark = True


manual = 'ddpm-pretrained-butterflies-john'

lr = 1e-5
timesteps = 500
scheduler = 'ddpm' # ddpm or ddim
beta_schedule = 'squaredcos_cap_v2' # ['linear', 'scaled_linear', 'squaredcos_cap_v2', 'sigmoid']

lr_f = f"{lr:.0e}".replace("e-0","e-").replace("e+0","e")

batch_size = 16

pipeline = DDPMPipeline.from_pretrained(f"Saved Models/ddpm-finetuned-{scheduler}-{beta_schedule}-{timesteps}-{lr_f}-{scheduler}" if not manual else manual).to(device) #

output = pipeline(num_inference_steps=timesteps, batch_size=batch_size).images
output_path = './Generations/' + (f'ddpm-finetuned-{scheduler}-{beta_schedule}-{timesteps}-{lr_f}-{scheduler}/' if not manual else manual+'/')

if not os.path.exists(output_path):
    os.mkdir(output_path)

gen = [torchvision.transforms.PILToTensor()(i).to(dtype=torch.float) for i in output]

torchvision.utils.save_image(gen, output_path+f'generatedimages.png', normalize=True)