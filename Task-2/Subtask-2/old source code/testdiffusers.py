from diffusers import DDIMScheduler, DDPMPipeline, DDPMScheduler, UNet2DModel


image_pipe = DDPMPipeline.from_pretrained('ddpm-pretrained-butterflies-john')
# image_pipe.save_pretrained('./ddpm-pretrained-butterflies-john')
# image_pipe.from_pretrained("ddpm-pretrained", use_safetensors=True).to('cuda')

config = image_pipe.unet.config
config.down_block_types[0] = 'AttnDownBlock2D'
config.down_block_types[1] = 'AttnDownBlock2D'
config.up_block_types[2] = 'AttnUpBlock2D'
config.up_block_types[3] = 'AttnUpBlock2D'


new_unet = UNet2DModel(**config)
old_state_dict = image_pipe.unet.state_dict()
new_unet.load_state_dict(old_state_dict, strict=False)

for k,v in config.items():
    print(k,':',v)