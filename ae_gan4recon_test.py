import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(0)

def read_data(file_path):
    column_names = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
    data = pd.read_csv(file_path, sep=' ', header = None, names = column_names)
    return data

def load_data(dataset, user_id, window_length=100, axis_num=9, data_path='./harbox_sensys21_dataset/'):
    class_set = ['Call','Hop','typing','Walk','Wave']
    class_num = len(class_set)
    
    for class_id in range(class_num):
        read_path = data_path +  str(user_id) + '/' + str(class_set[class_id]) + '_train' + '.txt'
        if os.path.exists(read_path):
            tmp_original_data = read_data(read_path)
            tmp_coll = tmp_original_data.iloc[:, 1:axis_num+1].to_numpy().reshape(-1, window_length, axis_num)
            if class_id not in dataset.keys():
                dataset[class_id]=[]
            dataset[class_id].extend(tmp_coll)
    return dataset

def split_dataset(dataset={}, test_size=0.2, window_length=100):
    for user_id in range(1,121):
        load_data(dataset, user_id, window_length=window_length)

    x_dataset = []
    y_dataset = []

    for key in dataset.keys():
        x_dataset.extend(dataset[key])
        y_dataset.extend(key*np.ones(len(dataset[key])).astype(int))

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size = test_size, random_state=42)
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split_dataset(test_size = 0.25, window_length=100)


class Autoencoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(Autoencoder,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.encoder = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding), nn.BatchNorm2d(out_channel)) # not good effect
        self.encoder = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(out_channel, in_channel, kernel_size, stride, padding))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def extract_feature_1(self, data, batch_size, window_length, device):
        result = []
        for idx in range(0, len(data)-batch_size, batch_size):
            sample = torch.tensor(data[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            sample = sample.to(device)
            tmp_x = self.encoder(sample)
            result.append(tmp_x)
        return result
    def extract_feature_2(self, data):
        result = []
        for idx in range(len(data)):
            tmp_x = self.encoder(data[idx])
            result.append(tmp_x)
        return result            

def autoencoder_train(model, loss_function, optimizer, x_train, y_train, window_length=100, batch_size=32, epochs=1, device='cpu'):
    model.train()
    for epoch in range(epochs+1):
        # correct = 0
        # batch_cnt=0
        for idx in range(0, len(x_train)-batch_size, batch_size):
            sample = torch.tensor(x_train[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            # true_scores = torch.tensor([y_train[idx:idx+batch_size]], dtype=torch.long).reshape(batch_size)
            sample = sample.to(device)
            # true_scores = true_scores.to(device)
            model.zero_grad()
            recon_sample = model(sample)
            loss = loss_function(recon_sample, sample)
            loss.backward()
            optimizer.step()
            
            # pred = tag_scores.argmax(dim=1, keepdim=True)
            # correct += pred.eq(true_scores.view_as(pred)).sum().item()
            # batch_cnt+=1
        clear_output(wait=True)
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()/batch_size))
    return model

def decoder_train(decoder_model, optimizer, data, raw_data, epochs=1, device='cpu'):
    model.train()    
    for epoch in range(epochs+1):
        for idx in range(len(data)):
            sample = data[idx]
            sample = sample.to(device)
            model.zero_grad()
            recon_sample = decoder_model(sample)
            loss = nn.MSELoss()(recon_sample, raw_data) # grad_penality
            loss.backward(retain_graph=True)
            optimizer.step()
        clear_output(wait=True)
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()/batch_size))
    return model


def autoencoder_train_2(model, loss_function, optimizer, data, window_length=100, batch_size=32, epochs=1, device='cpu'):
    model.train()
    for epoch in range(epochs+1):
        for idx in range(len(data)):
            sample = data[idx]
            sample = sample.to(device)
            model.zero_grad()
            recon_sample = model(sample)
            loss = loss_function(recon_sample, sample)
            loss.backward(retain_graph=True)
            optimizer.step()
        clear_output(wait=True)
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()/batch_size))
    return model


# generator
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size_1, kernel_size_2, stride=1, padding=0):
        super(Decoder, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size_1, stride, padding)
        self.conv_trans2 = nn.ConvTranspose2d(out_channel, 1, kernel_size_2, stride, padding) 
    def forward(self, x):
        x = self.conv_trans1(x)
        x = self.conv_trans2(x)
        return x


# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))

if __name__ == "__main__":

    axis_num=9
    batch_size=32
    epochs=150
    window_length=100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(1, 16, 3).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=5*1e-5)
    loss_function = nn.MSELoss()

    trained_model = autoencoder_train(model, loss_function, optimizer, x_train, y_train, window_length, batch_size, epochs, device)

    batch_size = 1
    sample = x_test[0:batch_size, :, :]
    sample_reshape = torch.tensor(sample).reshape(batch_size, 1, window_length, -1).float().to(device)
    with torch.no_grad():
        recon_sample = trained_model(sample_reshape)
    recon_sample_reshape = recon_sample.reshape(batch_size, window_length, -1)

    sample_one = sample[0]
    recon_sample_reshape_one = recon_sample_reshape[0].cpu().numpy()


    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (20,10)

    from scipy.interpolate import interp1d

    # def inter_func(idx):
    #     def scale(val):
    #         return (val - min(recon_sample_reshape_one[:, idx])) / (max(recon_sample_reshape_one[:, idx])\
    #                                                                 -min(recon_sample_reshape_one[:,idx])) * (max(sample_one[:, idx])-min(sample_one[:,idx])) + min(sample_one[:,idx])
    #     return scale

    def inter_func(idx):
        return interp1d([min(recon_sample_reshape_one[:, idx]), max(recon_sample_reshape_one[:, idx])], [min(sample_one[:, idx]), max(sample_one[:, idx])], kind = 'slinear', fill_value = 'extrapolate')

    x = np.arange(0, window_length)
    plt.subplot(3, 3, 1)
    plt.plot(x, sample_one[:, 0], '.-')
    plt.ylabel('X acceleration')

    tt = recon_sample_reshape_one[:, 0]
    # tt = inter_func(0)(recon_sample_reshape_one[:, 0])
    plt.plot(x, tt, 'c.-')
    plt.ylabel('X acceleration')

    plt.subplot(3, 3, 2)
    plt.plot(x, sample_one[:, 1], '.-')
    plt.ylabel('Y acceleration')

    tt = recon_sample_reshape_one[:, 1]
    # tt = inter_func(1)(recon_sample_reshape_one[:, 1])
    plt.plot(x, tt, 'c.-')
    plt.ylabel('Y acceleration')

    plt.subplot(3, 3, 3)
    plt.plot(x, sample_one[:, 2], '.-')
    plt.ylabel('Z acceleration')

    tt = recon_sample_reshape_one[:, 2]
    # tt = inter_func(2)(recon_sample_reshape_one[:, 2])
    plt.plot(x, tt, 'c.-')
    plt.ylabel('Z acceleration')


    plt.subplot(3, 3, 4)
    plt.plot(x, sample_one[:, 3], '.-')
    plt.ylabel('X acceleration')

    tt = recon_sample_reshape_one[:, 3]
    # tt = inter_func(3)(recon_sample_reshape_one[:, 3])
    plt.plot(x, tt, 'g.-')
    plt.ylabel('X acceleration')

    plt.subplot(3, 3, 5)
    plt.plot(x, sample_one[:, 4], '.-')
    plt.ylabel('Y acceleration')

    tt = recon_sample_reshape_one[:, 4]
    # tt = inter_func(4)(recon_sample_reshape_one[:, 4])
    plt.plot(x, tt, 'g.-')
    plt.ylabel('Y acceleration')

    plt.subplot(3, 3, 6)
    plt.plot(x, sample_one[:, 5], '.-')
    plt.ylabel('Z acceleration')

    tt = recon_sample_reshape_one[:, 5]
    # tt = inter_func(5)(recon_sample_reshape_one[:, 5])
    plt.plot(x, tt, 'g.-')
    plt.ylabel('Z acceleration')

    plt.subplot(3, 3, 7)
    plt.plot(x, sample_one[:, 6], '.-')
    plt.ylabel('X acceleration')

    tt = recon_sample_reshape_one[:, 6]
    plt.plot(x, tt, 'r.-')
    plt.ylabel('X-rec acceleration')

    plt.subplot(3, 3, 8)
    plt.plot(x, sample_one[:, 7], '.-')
    plt.ylabel('Y acceleration')

    tt = recon_sample_reshape_one[:, 7]
    # tt = inter_func(7)(recon_sample_reshape_one[:, 7])
    plt.plot(x, tt, 'r.-')
    plt.ylabel('Y acceleration')

    plt.subplot(3, 3, 9)
    plt.plot(x, sample_one[:, 8], '.-')
    plt.ylabel('Z acceleration')

    tt = recon_sample_reshape_one[:, 8]
    # tt = inter_func(8)(recon_sample_reshape_one[:, 8])
    plt.plot(x, tt, 'r.-')
    plt.ylabel('Z acceleration')

    plt.show()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_1 = Autoencoder(1, 16, 3).to(device)
    optimizer = optim.SGD(model_1.parameters(), lr=5*1e-5)
    loss_function = nn.MSELoss()

    trained_model_1 = autoencoder_train(model_1, loss_function, optimizer, x_train, y_train, window_length, batch_size, epochs, device)
    layer_1 = trained_model_1.extract_feature_1(x_train, batch_size, window_length, device)
    print(np.shape(layer_1[0]))


    model_2 = Autoencoder(16, 32, 3).to(device)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=5*1e-5)
    trained_model_2 = autoencoder_train_2(model_2, loss_function, optimizer_2, layer_1, window_length, batch_size, epochs, device)


    layer_2 = trained_model_2.extract_feature_2(layer_1)
    print(np.shape(layer_2[0]))


    model_3 = Autoencoder(32, 64, 3).to(device)
    optimizer_3 = optim.SGD(model_3.parameters(), lr=5*1e-5)
    trained_model_3 = autoencoder_train_2(model_3, loss_function, optimizer_3, layer_2, window_length, batch_size, epochs, device)


    layer_3 = trained_model_3.extract_feature_2(layer_2)
    print(np.shape(layer_3[0]))


    decoder = Decoder(64, 32, kernel_size_1=3, kernel_size_2=5, stride=1).to(device)
    print(np.shape(layer_3[0]))
    decode_output = decoder(layer_3[0])
    print(np.shape(decode_output))
