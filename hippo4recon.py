import time
import inspect
import random, os
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in [5, 6, 7])

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline, UNet2DModel, UNet2DConditionModel

from tqdm.auto import tqdm
from dataclasses import dataclass
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Union

seed = 42
n_classes = 5
data_path = './harbox_sensys21_dataset/'
data_save_folder = './backup/harbox/'

@dataclass
class TrainingConfig:
    data_size = (3, 100)
    train_batch_size = 128
    test_batch_size = 128
    ae_num_epochs = 100
    dm_num_epochs = 200
    num_train_timesteps = 1000
    num_inference_steps = 100
    gradient_accumulation_steps = 1
    learning_rate = 5e-4
    lr_warmup_steps = 10
    eta = 0
    train_size = 0.99
    channel_list = [64, 32, 16]
    kernel_size = 3
    window_length = 100
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = data_save_folder+'hippo_harbox'  # the model namy locally and on the HF Hub
    IMUD_norm = False # IMUD normalized--not good
    splitD_norm = True
    
    autoencoder_train = True # whether to re-train the model *****************************************
    layer_num = 3 # the i-th layer of extracted feature embedding *************************************
    corss_attention_dim = 3  # here needs to change according to the layer number :1-7, 2-5, 3-3 ******
    layer_embedding_name = data_save_folder+'layer_embedding_'+str(layer_num)+'.pt' # the saved embedding feature file *******************
    save_model_name = data_save_folder+'scheduler_model_l'+str(layer_num)+'.pt' # the saved diffusion model file ************************

config = TrainingConfig()

# Initialize accelerator and tensorboard logging
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps, 
    log_with="tensorboard",
    logging_dir=os.path.join(config.output_dir, "logs"),    
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = accelerator.device
generator_fix = torch.Generator(device=device)
generator_fix.manual_seed(seed)

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(Encoder,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.encoder = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
        self.pooling = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(out_channel, in_channel, kernel_size, stride, padding))
        self.unpooling = nn.MaxUnpool2d(kernel_size, stride, padding)

    def forward(self,x):
        x = self.encoder(x)
        x, indices = self.pooling(x)
        x = self.unpooling(x, indices) # filter out part of information since non-maximal values are lost
        x = self.decoder(x)
        return x

    def extract_feature_raw(self, x_data, batch_size, window_length, device):
        result = []
        batched_x_data = []
        batched_y_data = [] # we don't need the y_info, return []
        for idx in range(0, len(x_data)-batch_size, batch_size):
            sample = torch.tensor(x_data[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            batched_x_data.append(sample)
            sample = sample.to(device)
            sample_embedding = self.encoder(sample)
            result.append(sample_embedding)
        return result, batched_x_data, batched_y_data

    def extract_feature_embed(self, data):
        result = []
        for idx in range(len(data)):
            sample_embedding = self.encoder(data[idx])
            result.append(sample_embedding)
        return result

class Decoder(nn.Module):
    # minmize the error between raw data and reconstructed data
    def __init__(self, in_channel, out_channel, kernel_size_1, kernel_size_2, stride=1, padding=0):
        super(Decoder, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size_1, strFalseide, padding)
        self.conv_trans2 = nn.ConvTranspose2d(out_channel, 1, kernel_size_2, stride, padding)

    def forward(self, x):
        x = self.conv_trans1(x)
        x = self.conv_trans2(x)
        return x

class ExternalCrossAttention(nn.Module):
    def __init__(self, d_model,S=64):
        super().__init__()
        self.mq = nn.Linear(d_model, S, bias=False)
        self.mk = nn.Linear(d_model,S,bias=False)
        self.mv = nn.Linear(S,d_model,bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, latent_unet, latent_condition):
        Q = self.mq(latent_unet)
        K = self.mk(latent_condition)
        V = self.mv(latent_condition)
        attn=torch.bmm(Q, K)
        attn=self.softmax(attn)
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=torch.bmm(attn, V)
        return out

class UNet_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    def forward():
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class UNet_Encoder(nn.Module):
    def __init__(self, chs=(1, 8, 16, 32)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([UNet_Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)
    def forward():
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class UNet_Decoder(nn.Module):
    def __init__(self, chs = (32, 16, 8), cond_embed = None):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([UNet_Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.crossAttention = ExternalCrossAttention(d_model = 32, S = 128)
    def forward(self, x, encoder_features, embed_condition):
        x = self.crossAttention(x, embed_condition)
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim = 1) # note here is cat., so channel doubled
            x = self.dec_blocks[i](x)
        return x
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
                                      
class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 8, 16, 32), dec_chs=(32, 16, 8), num_class = 5, retrain_dim = False, out_sz = (1, 100)):
        super().__init__()
        self.encoder = UNet_Encoder(enc_chs)
        self.decoder = UNet_Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
    
    def forward(self, x, embed_condition):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:], embed_condition)
        # out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        return out

class IMUDataset(Dataset):
    def __init__(self, rootpath = data_path, saved = False, normalize = True):
        super().__init__()
        self.X = []
        self.y = []
        if saved == False:
            dataset_dict = load_data()
        for key in dataset_dict.keys():
            self.X.extend(dataset_dict[key])
            self.y.extend([int(key)]*np.ones(len(dataset_dict[key])).astype(int))
        if normalize == True:
            v = np.array(self.X)
            v_min = v.min(axis=(0, 1), keepdims=True)
            v_max = v.max(axis=(0, 1), keepdims=True)
            self.X = 2*(v - v_min)/(v_max - v_min)-1
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx])
        sample = sample[None,:]
        return sample, self.y[idx]

def read_data(file_path):
    column_names = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
    data = pd.read_csv(file_path, sep=' ', header = None, names = column_names)
    return data

def load_data(window_length=100, axis_num=9, data_path=data_path):
    dataset = {}
    class_set = ['Call','Hop','typing','Walk','Wave']
    class_num = len(class_set)
    for user_id in range(0, 121): #!(0, 121)
        for class_id in range(class_num):
            read_path = data_path +  str(user_id) + '/' + str(class_set[class_id]) + '_train' + '.txt'
            if os.path.exists(read_path):
                tmp_original_data = read_data(read_path)
                tmp_coll = tmp_original_data.iloc[:, 1:axis_num+1].to_numpy().reshape(-1, window_length, axis_num)
                if class_id not in dataset.keys():
                    dataset[class_id]=[]
                dataset[class_id].extend(tmp_coll)
    return dataset

def set_loader(config):
    imu_data = IMUDataset(rootpath = data_path, saved = False, normalize = config.IMUD_norm)
    train_len = int(len(imu_data)*config.train_size)
    test_len = len(imu_data)-train_len
    train_set, test_set = torch.utils.data.random_split(imu_data, [train_len, test_len])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.train_batch_size, num_workers=16, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.test_batch_size, num_workers=16, pin_memory=True, drop_last=True)
    return train_loader, test_loader

def split_dataset(train_size=0.8, window_length=100, normalize = True):
    dataset = load_data(window_length=window_length)
    test_size = 1 - train_size
    x_dataset = []
    y_dataset = []

    for key in dataset.keys():
        x_dataset.extend(dataset[key])
        y_dataset.extend(key*np.ones(len(dataset[key])).astype(int))
    
    if normalize == True:
        v = np.array(x_dataset)
        v_min = v.min(axis=(0, 1), keepdims=True)
        v_max = v.max(axis=(0, 1), keepdims=True)
        x_dataset = 2*(v - v_min)/(v_max - v_min)-1

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size = test_size, random_state=seed)
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    return x_train, x_test, y_train, y_test

def autoencoder_train_raw(model, loss_function, optimizer, x_train, window_length=100, batch_size=32, epochs=1, device='cpu'):
    model.train()
    print("******* Autoencoder train raw *******")
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

def autoencoder_train_embed(model, loss_function, optimizer, data, window_length=100, batch_size=32, epochs=1, device='cpu'):
    model.train()
    print("******* Autoencoder train embed *******")
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

def create_latent(x_train, layer_num=3, channel_list=[32, 64, 128], kernel_size=3, window_length=100, batch_size=32, epochs=100, device='cpu', mode_train = False):    
    if mode_train == True:
        # stacked autoencoder training--first layer encoder
        model = Encoder(1, channel_list[0], kernel_size).to(device)
        optimizer = optim.SGD(model.parameters(), lr=1*1e-4)
        loss_function = nn.MSELoss()
        model = autoencoder_train_raw(model, loss_function, optimizer, x_train, window_length, batch_size, epochs, device)
        torch.save(model, data_save_folder+'autoencoder_raw.pt')
    else:
        model = torch.load(data_save_folder+'autoencoder_raw.pt', map_location=torch.device(device))

    layer_1, batched_x_data, batched_y_data = model.extract_feature_raw(x_train, batch_size, window_length, device)
    
    if layer_num == 1:
        return layer_1, batched_x_data, batched_y_data

    if mode_train == True:
        # stacked autoencoder training--second layer encoder
        model_2 = Encoder(channel_list[0], channel_list[1], kernel_size).to(device)
        optimizer_2 = optim.SGD(model_2.parameters(), lr=5*1e-4)
        loss_function = nn.MSELoss()
        model_2 = autoencoder_train_embed(model_2, loss_function, optimizer_2, layer_1, window_length, batch_size, epochs, device)
        torch.save(model, data_save_folder+'autoencoder_raw.pt')
        torch.save(model_2, data_save_folder+'autoencoder_sec.pt')
    else:
        model_2 = torch.load(data_save_folder+'autoencoder_sec.pt', map_location=torch.device(device))

    layer_2 = model_2.extract_feature_embed(layer_1)
    
    if layer_num == 2:
        return layer_2, batched_x_data, batched_y_data
   
    if mode_train == True:
        # stacked autoencoder training--third layer encoder
        model_3 = Encoder(channel_list[1], channel_list[2], kernel_size).to(device)
        optimizer_3 = optim.SGD(model_3.parameters(), lr=5*1e-4)
        loss_function = nn.MSELoss()
        model_3 = autoencoder_train_embed(model_3, loss_function, optimizer_3, layer_2, window_length, batch_size, epochs, device)
        torch.save(model, data_save_folder+'autoencoder_raw.pt')
        torch.save(model_2, data_save_folder+'autoencoder_sec.pt')
        torch.save(model_3, data_save_folder+'autoencoder_third.pt')
    else:
        model_3 = torch.load(data_save_folder+'autoencoder_third.pt', map_location=torch.device(device))

    layer_3 = model_3.extract_feature_embed(layer_2)

    if layer_num == 3:
        return layer_3, batched_x_data, batched_y_data

def train_loop(config, model, encoder_hidden_states, noise_scheduler, optimizer, train_dataloader, lr_scheduler, do_classifier_free_guidance = True, save_model_name= 'scheduler_model.pt'):
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
            noise = torch.randn(clean_data.shape, generator=generator_fix, device = device, dtype = torch.float32).to(device)
            noise = noise * noise_scheduler.init_noise_sigma
            bs = clean_data.shape[0]
            
            # Sample a random timestep for each image ---todo test if fix the timestep works better---
            timesteps = torch.randint(int(noise_scheduler.num_train_timesteps/10), noise_scheduler.num_train_timesteps, (bs,), device = device).long()
            noisy_images = noise_scheduler.add_noise(clean_data, noise, timesteps)
            
            with accelerator.accumulate(model):
                # print(np.shape(noisy_images), np.shape(timesteps))
                noisy_images = noisy_images.to(torch.float32)
                # print(np.shape(encoder_hidden_states[idx]))
                hidden_view = encoder_hidden_states[idx].view(*encoder_hidden_states[idx].shape[:1], -1, encoder_hidden_states[idx].shape[-1])
                # print(np.shape(noisy_images), np.shape(hidden_view))
                if do_classifier_free_guidance == True:
                    if random.uniform(0, 1) > 0.95:
                        hidden_view = 1e-5*torch.ones(hidden_view.size()).to(device)
                noise_pred = model(noisy_images, timesteps, hidden_view, return_dict = False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss, retain_graph=True)
                
                # accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step = global_step)
            global_step+=1
            
    save_object_dict = {"noise_scheduler": noise_scheduler, "model": accelerator.unwrap_model(model = model)}
    torch.save(save_object_dict, save_model_name)
    
    return noise_scheduler, model

@torch.no_grad()
def evaluate(latents_shape, scheduler, model, encoder_hidden_states, device, num_inference_steps = 1,  guidance_scale = 2, do_classifier_free_guidance = True):
    extra_step_kwargs = {}

    noise_latents = torch.randn(latents_shape, device=device)
    random_data = noise_latents
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.to(device)
    noise_latents = noise_latents*scheduler.init_noise_sigma
    
    hidden_view = encoder_hidden_states[0].view(*encoder_hidden_states[0].shape[:1], -1, encoder_hidden_states[0].shape[-1])
    uncond_view = 1e-5*torch.ones(hidden_view.size()).to(device)
    hidden_view = torch.cat([uncond_view, hidden_view])
    # print(np.shape(latent_model_input), np.shape(hidden_view))

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([noise_latents]*2) if do_classifier_free_guidance else noise_latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = model(latent_model_input, t, hidden_view).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        noise_latents = scheduler.step(noise_pred, t, noise_latents, **extra_step_kwargs).prev_sample
    return random_data, noise_latents

if __name__ == "__main__": 
   
    x_train, x_test, y_train, y_test = split_dataset(train_size = config.train_size, window_length = config.window_length, normalize = config.splitD_norm)
    latent_shape = (config.train_batch_size, 1, *np.shape(x_train[0]))
    # latent_shape = (1, 1, *np.shape(x_train[0]))
    
    layer_embedding, batched_data, true_scores = create_latent(x_train, config.layer_num, config.channel_list, config.kernel_size, config.window_length, config.train_batch_size, config.ae_num_epochs, device=device, mode_train=config.autoencoder_train)
    print(len(layer_embedding), layer_embedding[0].shape)
    torch.save(layer_embedding, config.layer_embedding_name)
    # layer_embedding = torch.load(config.layer_embedding_name)

    train_loader, test_loader = set_loader(config)
    
    # check the source file here https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py
    model = UNet2DConditionModel(
        sample_size=config.data_size,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=1,  # how many ResNet layers to use per UNet block
        # down_block_types = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        # up_block_types = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(32, 64, 128, 256),  # the number of output channes for each UNet block.
        cross_attention_dim=config.corss_attention_dim,
    )

    save_model_name = config.save_model_name
    noise_scheduler = DDIMScheduler(num_train_timesteps = config.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=(len(train_loader)*config.dm_num_epochs))
    
    noise_scheduler, model = train_loop(config, model, layer_embedding, noise_scheduler, optimizer, train_loader, lr_scheduler, save_model_name=save_model_name)

    saved_object_dict = torch.load(save_model_name, map_location=torch.device(device))
    noise_scheduler, model = saved_object_dict["noise_scheduler"], saved_object_dict["model"]
    begin=time.time()
    random_data, generated_data = evaluate(latent_shape, noise_scheduler, model, encoder_hidden_states = layer_embedding, device = device, num_inference_steps = config.num_inference_steps)
    end=time.time()
    print("Time spent: ", end-begin)
    print(np.shape(generated_data))
