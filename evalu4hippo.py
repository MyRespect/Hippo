import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in [4, 5, 6, 7])

import inspect
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh import extract_features
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d, uniform_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cae import CAE
from config import TrainingConfig
from hippo4recon import create_latent, evaluate
from harbox_dataset import IMUDataset, split_dataset, load_data, set_loader, load_data_with_id

config = TrainingConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
config = TrainingConfig()
window_length = config.window_length

random.seed(seed)
generator_fix = torch.Generator(device=device)
generator_fix.manual_seed(seed)

dataset = load_data_with_id(id_list = ['1', '50', '100'])

def draw_raw_data(dataset, window_length):
    plt.rcParams["figure.figsize"] = (15,4)
    fig, axes = plt.subplots(2, 2)

    x = np.arange(0, window_length)

    print("Five activities from 2 different users: ")
    for user_id in list(dataset.keys())[0:2]: # get the first two people
        for class_id in range(2):
            if user_id == '1':
                # dataset[str(user_id)][class_id][0][:,0]
                axes[0][class_id].plot(x, np.mean(dataset[str(user_id)][class_id][0][:,0:9], axis=1), '.-')
            else:
                axes[1][class_id].plot(x, np.mean(dataset[str(user_id)][class_id][0][:,0:9], axis=1), '.-')
    plt.show()

def draw_gen_data(random_data, generated_data, window_length):
    plt.rcParams["figure.figsize"] = (15, 6)
    n_axis = 9
    x = np.arange(0, window_length)
    fig, axes = plt.subplots(3, 5)

    for data_type in ['R', 'G', 'O']:
        for idx in range(5):
            if data_type == 'R':
                axes[0][idx].plot(x, torch.mean(random_data[idx][0][:,0:n_axis], dim=1), '.-')
            elif data_type =='G':
                gen_x = torch.mean(generated_data[idx][0][:,0:n_axis], dim=1)
                gen_xx = uniform_filter1d(gen_x, size=3)
                axes[1][idx].plot(x, gen_xx, '.-')  # consider if we need to filter out the data
            else:
                axes[2][idx].plot(x, np.mean(test_data[idx][:,0:n_axis], axis=1), '.-')
    plt.show()

def data_gen(config):
    for layer_num in range(1, 4):
        diffusion_model_file = config.output_dir+'scheduler_model_l'+str(layer_num)+'un.pt'
        config.save_model_loc = diffusion_model_file

        print("layer_num is " layer_num)
        batch_size = 16
        latent_shape = (batch_size, 1, config.window_length, 9)

        array_list_0 = []
        array_list_1 = []
        array_list_2 = []
        array_list_3 = []
        array_list_4 = []

        for nn in range(50):
            for i in range(5):
                test_data = np.array(dataset['1'][i][0:21]) ## the same activity

                layer_embedding, batched_data = create_latent(test_data, config)

                random_data, generated_data = evaluate(config, latent_shape, encoder_hidden_states = layer_embedding, device = device, num_inference_steps = 100)

                random_data = random_data.cpu()
                generated_data = generated_data.cpu()

                if i ==0:
                    array_list_0.append(generated_data.numpy())
                elif i ==1:
                    array_list_1.append(generated_data.numpy())
                elif i==2:
                    array_list_2.append(generated_data.numpy())
                elif i ==3:
                    array_list_3.append(generated_data.numpy())
                else:
                    array_list_4.append(generated_data.numpy())
        array_list_all = [array_list_0, array_list_1, array_list_2, array_list_3, array_list_4]
        save_data = np.array(array_list_all)
        print(np.shape(save_data))
        save_data = save_data.reshape(5, -1, config.window_length, 9)
        np.save('./contrastive_figures/draw_figures/contrastive_data_'+str(layer_num)+'.npy', save_data)

class CNNClassifier(nn.Module):
    def __init__(self, window_length, axis_num, label_num):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*(window_length-4)*(axis_num-4), 128)
        self.fc2 = nn.Linear(128, label_num)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def cnn_train(model, loss_function, optimizer, x_train, y_train, window_length=100, batch_size=32, epochs=1, device='cpu'):
    model.train()
    for epoch in range(epochs):
        correct = 0
        batch_cnt=0
        for idx in range(0, len(x_train)-batch_size, batch_size):
            sample = torch.tensor(x_train[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            true_scores = torch.tensor([y_train[idx:idx+batch_size]], dtype=torch.long).reshape(batch_size)
            sample = sample.to(device)
            true_scores = true_scores.to(device)
            model.zero_grad()
            tag_scores = model(sample)
            loss = loss_function(tag_scores, true_scores)
            loss.backward()
            optimizer.step()
            
            pred = tag_scores.argmax(dim=1, keepdim=True)
            correct += pred.eq(true_scores.view_as(pred)).sum().item()
            batch_cnt+=1
        clear_output(wait=True)
        print('Train Epoch: {} \tLoss: {:.6f} \t {}'.format(epoch, loss.item(), correct/(batch_cnt*batch_size)))
    return model

def cnn_test(model, loss_function, x_test, y_test, window_length=100, batch_size=32, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    batch_cnt=0
    len_test = len(x_test)
    with torch.no_grad():
        for idx in range(0, len_test-batch_size+1, batch_size):
            data=torch.tensor(x_test[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            true_scores = torch.tensor([y_test[idx:idx+batch_size]], dtype=torch.long).reshape(batch_size)
            data = data.to(device)
            true_scores = true_scores.to(device)
            # print(np.shape(data))
            output = model(data)
            # print(output)
            test_loss += loss_function(output, true_scores).sum().item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            correct += pred.eq(true_scores.view_as(pred)).sum().item()
            batch_cnt+=1

    test_loss /= (len_test-batch_size)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len_test, 100. * correct / (batch_cnt*batch_size)))

def save_gen_data(config):
    config.layer_num = 3
    saved_data_file = config.output_dir+'generated_data_dict_p1_l'+str(config.layer_num)+'.pt'
    diffusion_model_file = config.output_dir+'scheduler_model_l'+str(config.layer_num)+'un.pt'

    config.save_model_loc = diffusion_model_file

    latent_shape = (batch_size, 1, 100, 9)
    generated_data_dict={'0':[], '1':[], '2':[], '3':[], '4':[]} # 'Call','Hop','typing','Walk','Wave'

    for i in range(10):
        for activity_id in range(5):

            if i < 5:
                test_data = np.array(dataset['1'][activity_id][0:batch_size+1]) ## [ppl_id, activity, #of samples]: the same activity
            else:
                test_data = np.array(dataset['50'][activity_id][0:batch_size+1]) ## [ppl_id, activity, #of samples]: the same activity
            # print("np.shape(test_data)", np.shape(test_data)) # [batch_size+1, 100, 9], avoiding idx out range in evaluate()

            layer_embedding, batched_data = create_latent(test_data, config)

            random_data, generated_data = evaluate(config, latent_shape, encoder_hidden_states = layer_embedding, device = device, num_inference_steps = 200)

            random_data = random_data.cpu()
            generated_data = generated_data.cpu()
            generated_data = uniform_filter(generated_data, size=3) ### apply moving average smoothing
            generated_data_dict[str(activity_id)].append(generated_data)
        clear_output(wait=True)
        print('Round: ', i)

    torch.save(generated_data_dict, saved_data_file)

def load_gen_data(config):
    saved_data_file = config.output_dir+'generated_data_dict_p1_l'+str(config.layer_num)+'.pt'
    generated_data_dict = torch.load(saved_data_file)
    print(np.shape(generated_data_dict['0'][0])) # 1
    X_list = []
    Y_list = []
    for activity_id in range(5):
        for i in range(10):
            reshape_data = np.array(generated_data_dict[str(activity_id)][i].reshape(-1, 100, 9))
            X_list.append(reshape_data)
            Y_list.append(np.ones(batch_size, dtype=np.int8)*activity_id)

    XX=np.array(X_list).reshape(-1, 100, 9)
    YY=np.array(Y_list).reshape(-1, )
    print(np.shape(XX), np.shape(YY))
    return XX, YY

def apply_classifier_semantic(config):
    XX, YY = load_gen_data(config)
    x_train, x_test, y_train, y_test = train_test_split(XX, YY, test_size = 0.25, random_state=42)

    window_length=50
    axis_num=9
    batch_size=32
    epochs=50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNClassifier(window_length=window_length, axis_num=axis_num, label_num=5).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    trained_model = cnn_train(model, loss_function, optimizer, x_train, y_train, window_length, batch_size, epochs, device)

    loss_function = nn.NLLLoss()
    # trained_model = torch.load('cnn_harbox_model.pt', map_location=torch.device(device))

    print("Finish training and start testing: ")
    cnn_test(trained_model, loss_function, x_test, y_test, batch_size=batch_size, device=device)


@torch.no_grad() # metadate-level information filter
def filter_raw(train_dataloader, scheduler, model, encoder_hidden_states, device, num_inference_steps = 100,  guidance_scale = 7.5, do_classifier_free_guidance = True):
    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = 0
    filtered_X = []
    filtered_y = []
    
    for idx, (clean_data, y) in enumerate(train_dataloader):
        clean_data = clean_data.to(device)
        
        noise = torch.randn(clean_data.shape, generator=generator_fix, device = device, dtype = torch.float32)
        noise = noise * scheduler.init_noise_sigma
        
        bs = clean_data.shape[0]
        fwd_timesteps = torch.randint(1, num_inference_steps, (bs,), device = device).long()
        
        noise_latents = scheduler.add_noise(clean_data, noise, fwd_timesteps).to(torch.float32)
        noise_latents = noise_latents*scheduler.init_noise_sigma
    
        hidden_view = encoder_hidden_states[0].view(*encoder_hidden_states[0].shape[:1], -1, encoder_hidden_states[0].shape[-1])
        uncond_view = 1e-4*torch.ones(hidden_view.size()).to(device)
        hidden_view = torch.cat([uncond_view, hidden_view])
        # print(np.shape(latent_model_input), np.shape(hidden_view))

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps.to(device)
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([noise_latents]*2) if do_classifier_free_guidance else noise_latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = model(latent_model_input, t, hidden_view).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            noise_latents = scheduler.step(noise_pred, t, noise_latents, **extra_step_kwargs).prev_sample
        filtered_X.append(noise_latents)
        filtered_y.append(y)
        
    return filtered_X, filtered_y

def filter_meta_data():
    train_loader = set_loader()
    embedding_file = config.output_dir+'layer_embedding_1un.pt'
    diffusion_model_file = config.output_dir+'scheduler_model_l1un.pt'
    layer_embedding = torch.load(embedding_file, map_location=torch.device(device))
    saved_object_dict = torch.load(diffusion_model_file, map_location=torch.device(device))
    noise_scheduler, model = saved_object_dict["noise_scheduler"], saved_object_dict["model"]
    print(layer_embedding[0].shape)

    filtered_X, filtered_y = filter_raw(train_loader, noise_scheduler, model, layer_embedding, device)

    for i in range(len(filtered_X)):
        filtered_X[i]=filtered_X[i].detach().cpu()
        filtered_y[i]=filtered_y[i].detach().cpu()
    for i in range(len(filtered_X)-1):
        filtered_X[0] = np.concatenate((filtered_X[0], filtered_X[i+1]), axis=0)
        filtered_y[0] = np.concatenate((filtered_y[0], filtered_y[i+1]), axis=0)
    reshape_data = np.array(filtered_X[0].reshape(-1, 100, 9))
    reshape_y = np.array(filtered_y[0])
    print(np.shape(reshape_data))

    # x = np.arange(0, config.window_length)
    # fig, axes = plt.subplots(1, 5)

    # for class_id in range(5):
    #     axes[class_id].plot(x, np.mean(reshape_data[class_id][:,0:9], axis=1), '.-')
    # plt.show()

    return reshape_data, reshape_y

def apply_classifier_meta(config):
    x_train, x_test, y_train, y_test = train_test_split(reshape_data, reshape_y, test_size = 0.25, random_state=42)

    window_length=50
    axis_num=9
    batch_size=32
    epochs=50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNClassifier(window_length=window_length, axis_num=axis_num, label_num=5).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    trained_model = cnn_train(model, loss_function, optimizer, x_train, y_train, window_length, batch_size, epochs, device)

    cnn_test(trained_model, loss_function, x_test, y_test, batch_size=batch_size, device=device)

def statistic_comparison(config):
    saved_data_file_1 = config.output_dir+'generated_data_dict_p1_l'+str(1)+'.pt'
    saved_data_file_2 = config.output_dir+'generated_data_dict_p1_l'+str(2)+'.pt'

    generated_data_dict = torch.load(saved_data_file_1)
    reshape_data = np.array(generated_data_dict['0'][0].view(-1, 9) )
    data_df = pd.DataFrame(reshape_data, columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'])
    data_df["id"] = 1
    data_df['timestamp'] = range(0,reshape_data.shape[0])
    extracted_features = extract_features(data_df, column_id="id", column_sort="timestamp")
    feature_dict = extracted_features.to_dict()

    generated_data_dict_1 = torch.load(saved_data_file_2)
    reshape_data_1 = np.array(generated_data_dict_1['0'][0].view(-1, 9) )
    data_df_1 = pd.DataFrame(reshape_data_1, columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z'])
    data_df_1["id"] = 1
    data_df_1['timestamp'] = range(0,reshape_data_1.shape[0])
    extracted_features_1 = extract_features(data_df_1, column_id="id", column_sort="timestamp")
    feature_dict_1 = extracted_features_1.to_dict()

    from dictdiffer import diff, patch, swap, revert
    result = diff(feature_dict, feature_dict_1, tolerance=0.1)
    print(list(result))
