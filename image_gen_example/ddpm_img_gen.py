# https://github.com/huggingface/diffusers

from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

step = 2000

# automatically download files: ~/.cache/huggingface/diffusers/
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256") 
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)
scheduler.set_timesteps(step)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to(device)
model_input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(model_input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, model_input).prev_sample
        model_input = prev_noisy_sample

image = (model_input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
file_name = "dm_image_example_"+str(step)+".jpg"
image.save(file_name)