from dataclasses import dataclass

@dataclass
class TrainingConfig:
    data_size = (50, 9)
    batch_size = 16
    ae_num_epochs = 100
    dm_num_epochs = 50
    num_train_timesteps = 1000
    num_inference_steps = 100
    gradient_accumulation_steps = 1
    learning_rate = 5e-4
    lr_warmup_steps = 500
    eta = 1
    train_ratio = 1
    channel_list = [32, 16, 8]
    kernel_size = 3
    window_length = 50
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = './hippo_harbox/'  # the model namy locally and on the HF Hub
    data_scale = False
    
    autoencoder_train = [False, False, False] # whether to re-train the model
    layer_num = 3 # the i-th layer of extracted feature embedding
    corss_attention_dim = [7,5,3]  # here needs to change according to the layer number :1-7, 2-5, 3-3
    layer_embedding_loc = output_dir+'layer_'+str(layer_num)+'_embedding.pt' # the saved embedding feature file
    save_model_loc = output_dir+'dm_l'+str(layer_num)+'.pt' # the saved diffusion model file