import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in [3])

import time
import inspect
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

from cae import CAE
from harbox_dataset import IMUDataset, split_dataset, load_data, set_loader

from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline, UNet2DModel, UNet2DConditionModel

from tqdm.auto import tqdm
from config import TrainingConfig
from accelerate import Accelerator
from typing import List, Optional, Tuple, Union

config = TrainingConfig()

# Initialize accelerator and tensorboard logging
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps, 
    log_with="tensorboard",
    project_dir=os.path.join(config.output_dir, "logs"),    
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = accelerator.device
generator = torch.Generator(device=device)
generator.manual_seed(42)

def autoencoder_train_raw(model, x_train, window_length=100, batch_size=32, epochs=1, device='cpu'):
    optimizer = optim.SGD(model.parameters(), lr = 1e-4)
    loss_function = nn.MSELoss()    

    model.train()
    print("******* Train Stack-1 Autoencoder *******")
    for epoch in range(epochs+1):
        for idx in range(0, len(x_train)-batch_size, batch_size):
            sample = torch.tensor(x_train[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            sample = sample.to(device)
            model.zero_grad()
            recon_sample = model(sample)
            loss = loss_function(recon_sample, sample)
            loss.backward()
            optimizer.step()
        if epoch % 10  ==0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()/batch_size))
    return model

def autoencoder_train_embed(model, data, window_length=100, batch_size=32, epochs=1, device='cpu'):
    optimizer = optim.SGD(model.parameters(), lr = 1e-4)
    loss_function = nn.MSELoss()

    model.train()
    print("******* Train Stack-i Autoencoder *******")
    for epoch in range(epochs+1):
        for idx in range(len(data)):
            sample = data[idx]
            sample = sample.to(device)
            model.zero_grad()
            recon_sample = model(sample)
            loss = loss_function(recon_sample, sample)
            loss.backward(retain_graph=True)
            optimizer.step()
        if epoch % 10 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()/batch_size))
    return model

def create_latent(x_train, config, device=device):    
    batch_size = config.batch_size
    mode_train=config.autoencoder_train
    epochs = config.ae_num_epochs

    if mode_train[0] == True:
        # Stack-1 CAE
        model = CAE(1, config.channel_list[0], config.kernel_size).to(device)
        model = autoencoder_train_raw(model, x_train, config.window_length, batch_size, epochs, device)
        torch.save(model, config.output_dir+'cae_stack1.pt')
    else:
        model = torch.load(config.output_dir+'cae_stack1.pt', map_location=torch.device(device))

    layer_1, batched_x_data = model.extract_feature_raw(x_train, batch_size, config.window_length, device)
    
    if config.layer_num == 1:
        return layer_1, batched_x_data

    layer_embed = layer_1
    for i in range(1, config.layer_num):
        if mode_train[i] == True:
            # Stack-i CAE
            model = CAE(config.channel_list[i-1], config.channel_list[i], config.kernel_size).to(device)
            model = autoencoder_train_embed(model, layer_embed, config.window_length, batch_size, epochs, device)
            torch.save(model, config.output_dir+'cae_stack'+str(i+1)+'.pt')
        else:
            model = torch.load(config.output_dir+'cae_stack'+str(i+1)+'.pt', map_location=torch.device(device))

        layer_i = model.extract_feature_embed(layer_embed)
        layer_embed = layer_i
    
    return layer_embed, batched_x_data

def train_loop(config, model, encoder_hidden_states, noise_scheduler, optimizer, train_dataloader, lr_scheduler, do_classifier_free_guidance = True, save_model_loc= 'scheduler_model.pt'):
    # global varabile: accelerator
    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    global_step = 0
    accepts_eta = "eta" in set(inspect.signature(noise_scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = config.eta
    
    for epoch in range(config.dm_num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for idx, (clean_data, y) in enumerate(train_dataloader):
            noise = torch.randn(clean_data.shape, generator=generator, device = device, dtype = torch.float32).to(device)
            noise = noise * noise_scheduler.init_noise_sigma
            bs = clean_data.shape[0]

            # Sample a random timestep for each image ---todo test if fix the timestep works better---
            timesteps = torch.randint(int(noise_scheduler.config.num_train_timesteps/10), noise_scheduler.config.num_train_timesteps, (bs,), device = device).long()
            noisy_images = noise_scheduler.add_noise(clean_data, noise, timesteps)
            
            with accelerator.accumulate(model):
                # print(np.shape(noisy_images), np.shape(timesteps))
                noisy_images = noisy_images.to(torch.float32)
                # print(np.shape(encoder_hidden_states[idx]))
                hidden_view = encoder_hidden_states[idx].view(encoder_hidden_states[idx].shape[0], -1, encoder_hidden_states[idx].shape[-1])
                # print(np.shape(noisy_images), np.shape(hidden_view))
                if do_classifier_free_guidance == True:
                    if random.uniform(0, 1) > 0.95:
                        hidden_view = 1e-5*torch.ones(hidden_view.size()).to(device)

                noise_pred = model(noisy_images, timesteps, hidden_view, return_dict = False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss, retain_graph=True)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step = global_step)
            global_step+=1
            
    save_object_dict = {"noise_scheduler": noise_scheduler, "model": accelerator.unwrap_model(model = model)}
    torch.save(save_object_dict, config.save_model_loc)

@torch.no_grad()
def evaluate(config, latents_shape, encoder_hidden_states, device, num_inference_steps = 1,  guidance_scale = 7.5, do_classifier_free_guidance = True):
    extra_step_kwargs = {}

    saved_object_dict = torch.load(config.save_model_loc, map_location=torch.device(device))
    scheduler, model = saved_object_dict["noise_scheduler"], saved_object_dict["model"]

    noise_latents = torch.randn(latents_shape, device=device)
    random_data = noise_latents
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.to(device)
    noise_latents = noise_latents*scheduler.init_noise_sigma
    
    hidden_view = encoder_hidden_states[0].view(encoder_hidden_states[0].shape[0], -1, encoder_hidden_states[0].shape[-1])
    uncond_view = 1e-5*torch.ones(hidden_view.size()).to(device)
    hidden_view = torch.cat([uncond_view, hidden_view])
    # print(np.shape(latent_model_input), np.shape(hidden_view))

    begin=time.time()

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([noise_latents]*2) if do_classifier_free_guidance else noise_latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = model(latent_model_input, t, hidden_view).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        noise_latents = scheduler.step(noise_pred, t, noise_latents, **extra_step_kwargs).prev_sample

    end=time.time()
    print("Time spent: ", end-begin)

    # import matplotlib.pyplot as plt
    # for i in range(2, 16, 2):
    #     output_signal = noise_latents[i][0].cpu().numpy()
    #     avg3_val = np.sum(output_signal[:,0:3], axis = 1)/3.
    #     plt.plot(range(avg3_val.shape[0]), avg3_val)
    #     plt.savefig('accelerometer_'+str(i)+'.png')

    return random_data, noise_latents

if __name__ == "__main__":

    train_dm = True

    latent_shape = (config.batch_size, 1, config.window_length, 9)

    if train_dm == True:

        x_train, x_test, y_train, y_test = split_dataset(train_size = config.train_ratio, window_length = config.window_length, scale = config.data_scale)
                
        layer_embedding, batched_data = create_latent(x_train, config, device=device)
        torch.save(layer_embedding, config.layer_embedding_loc)

        train_loader, test_loader = set_loader(config)
        
        # check the source file here https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py
        model = UNet2DConditionModel(
            sample_size=config.data_size,  # the target data resolution
            in_channels=1,  # the number of input channels
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64, 128),  # the number of output channes for each UNet block. Min_channel is 32 in the UNet
            cross_attention_dim=config.corss_attention_dim[config.layer_num-1],
        )

        noise_scheduler = DDIMScheduler(clip_sample = False, beta_start = 0.00056, beta_end = 0.0115)
        noise_scheduler.init_noise_sigma = 11.5 # about self.sigmas.max()
        noise_scheduler.config.num_train_timesteps = config.num_train_timesteps
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=(len(train_loader)*config.dm_num_epochs))
        
        train_loop(config, model, layer_embedding, noise_scheduler, optimizer, train_loader, lr_scheduler, save_model_loc=config.save_model_loc)
    else:
        print("Generate new data from random noise: ")
        layer_embedding = torch.load(config.layer_embedding_loc)
        evaluate(config, latent_shape, encoder_hidden_states = layer_embedding, device = device, num_inference_steps = config.num_inference_steps)

