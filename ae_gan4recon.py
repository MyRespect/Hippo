# Implement of Autoencoder for Data Reconstruction

import os, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from sklearn.model_selection import train_test_split

torch.autograd.set_detect_anomaly(True)

seed = 42
n_classes = 5
data_path = './harbox_sensys21_dataset/'
saved_data_path = './ae4recon_files/'

random.seed(seed)

def read_data(file_path):
    column_names = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
    data = pd.read_csv(file_path, sep=' ', header = None, names = column_names)
    return data

def load_data(dataset, user_id, window_length=100, axis_num=9, data_path=data_path):
    class_set = ['Call','Hop','typing','Walk','Wave'] # the name of files
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

def split_dataset(dataset={}, test_size=0.3, window_length=100):
    for user_id in range(0, 1): # (0, 121)
        load_data(dataset, user_id, window_length=window_length)

    x_dataset = []
    y_dataset = []

    for key in dataset.keys():
        x_dataset.extend(dataset[key])
        y_dataset.extend(key*np.ones(len(dataset[key])).astype(int))

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size = test_size, random_state=seed)
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    return x_train, x_test, y_train, y_test

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

    def extract_feature_raw(self, x_data, y_data, batch_size, window_length, device):
        result = []
        batched_x_data = []
        batched_y_data = []
        for idx in range(0, len(x_data)-batch_size, batch_size):
            sample = torch.tensor(x_data[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            batched_y_data.append(torch.tensor(y_data[idx:idx+batch_size]))
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
    def __init__(self, in_channel, out_channel, kernel_size_1, kernel_size_2, stride=1, padding=0):
        super(Decoder, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size_1, stride, padding)
        self.conv_trans2 = nn.ConvTranspose2d(out_channel, 1, kernel_size_2, stride, padding)

    def forward(self, x):
        x = self.conv_trans1(x)
        x = self.conv_trans2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, window_length, axis_num, label_num):
        super(Discriminator, self).__init__()
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

class Discriminator_2(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size):
        super(Discriminator_2, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first = True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, label_size)

    def forward(self, embeds):
        lstm_out, _ = self.lstm(embeds)
        tag_scores = self.hidden2tag(lstm_out)
        # tag_scores = F.log_softmax(tag_scores, dim=0)
        return tag_scores


def save_model(model, filename):
    print('Saving the model.state_dict')
    torch.save(model.state_dict(), filename)

def get_categorical(labels, n_classes=n_classes):
    cat = labels
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)

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

def create_latent(x_train, y_train, layer_num=3, window_length=100, batch_size=32, epochs=100, device='cpu'):

    # generate the hierarchical latent representations

    # stacked autoencoder training--first layer encoder
    model = Encoder(1, 128, 3).to(device)
    optimizer = optim.SGD(model.parameters(), lr=5*1e-5)
    loss_function = nn.MSELoss()

    autoencoder_train_raw(model, loss_function, optimizer, x_train, window_length, batch_size, epochs, device)
    layer_1, batched_x_data, batched_y_data = model.extract_feature_raw(x_train, y_train, batch_size, window_length, device)

    if layer_num == 1:
        return layer_1, batched_x_data, batched_y_data

    # stacked autoencoder training--second layer encoder
    model_2 = Encoder(128, 64, 3).to(device)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=5*1e-5)
    loss_function = nn.MSELoss()

    autoencoder_train_embed(model_2, loss_function, optimizer_2, layer_1, window_length, batch_size, epochs, device)
    layer_2 = model_2.extract_feature_embed(layer_1)

    if layer_num == 2:
        return layer_2, batched_x_data, batched_y_data

    # stacked autoencoder training--third layer encoder
    model_3 = Encoder(64, 32, 3).to(device)
    optimizer_3 = optim.SGD(model_3.parameters(), lr=5*1e-5)
    loss_function = nn.MSELoss()

    autoencoder_train_embed(model_3, loss_function, optimizer_3, layer_2, window_length, batch_size, epochs, device)
    layer_3 = model_3.extract_feature_embed(layer_2)

    if layer_num == 3:
        return layer_3, batched_x_data, batched_y_data

def adversarial_train(raw_x_data, raw_y_data, layer_num=3, window_length=100, batch_size=32, epochs=100, device='cpu'):

    layer_embedding, batched_data, true_scores = create_latent(raw_x_data, raw_y_data, layer_num, window_length, batch_size, epochs, device)

    if layer_num == 1:
        decoder = Decoder(128, 64, kernel_size_1 = 3, kernel_size_2 = 1, stride = 1).to(device)
    if layer_num == 2:
        decoder = Decoder(64, 32, kernel_size_1 = 3, kernel_size_2 = 3, stride = 1).to(device)
    if layer_num == 3:
        decoder = Decoder(32, 16, kernel_size_1 = 3, kernel_size_2 = 5, stride = 1).to(device)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)
    decoder_loss_func = nn.MSELoss()

    # discriminator = Discriminator(100, 9, 5).to(device)
    # discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=0.1)
    # discriminator_loss_func = nn.NLLLoss()

    discriminator = Discriminator_2(9, 100, 5).to(device)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0007)
    discriminator_loss_func = nn.CrossEntropyLoss()

    decoder.train()
    discriminator.train()

    for epoch in range(epochs):
        for i_batch in range(len(layer_embedding)):
            item_data = layer_embedding[i_batch].to(device)
            item_raw_data = batched_data[i_batch].to(device)
            item_raw_y = true_scores[i_batch].to(device)

            # decoder loss

            decoder.zero_grad()
            discriminator.zero_grad()

            decode_output = decoder(item_data)
            decoder_mse_loss = decoder_loss_func(decode_output, item_raw_data)

            eps = Variable(torch.rand(batch_size, 1, 100, 9), requires_grad=True).to(device)

            X_inter = eps*item_raw_data+(1-eps)*decode_output
            X_inter_reshape = torch.reshape(X_inter, (-1, 100, 9))
            X_inter_output_reshape = Variable(X_inter_reshape, requires_grad=True)
            X_inter_grad = torch.autograd.grad(torch.sum(discriminator(X_inter_output_reshape)[:, -1, :]), X_inter_output_reshape)[0]
            X_inter_grad_norm = torch.sqrt(torch.sum(X_inter_grad**2, dim=1))
            grad_pen = 0.1*torch.mean(torch.nn.ReLU()(X_inter_grad_norm-1))


            # decode_tag_scores = discriminator(decode_output)

            decode_output_reshape = torch.reshape(decode_output, (-1, 100, 9))
            discriminator_output = discriminator(decode_output_reshape)
            decode_tag_scores = discriminator_output[:, -1, :]

            disc_decode_loss = discriminator_loss_func(decode_tag_scores, item_raw_y)+grad_pen

            decoder_loss = decoder_mse_loss + disc_decode_loss

            decoder_loss.backward(retain_graph=True)
            decoder_optimizer.step()

            decoder.zero_grad()
            discriminator.zero_grad()

            # raw_tag_scores = discriminator(item_raw_data)

            decode_raw_reshape = torch.reshape(item_raw_data, (-1, 100, 9))
            discriminator_raw_output = discriminator(decode_raw_reshape)
            raw_tag_scores = discriminator_raw_output[:, -1, :]

            disc_raw_loss = discriminator_loss_func(raw_tag_scores, item_raw_y)

            disc_decode_loss_detach = disc_decode_loss.detach()
            discriminator_loss = torch.mean(torch.log(disc_raw_loss) + torch.log(1 - disc_decode_loss_detach))

            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()    

    return decoder, discriminator

def model_test(decoder, discriminator, x_test, y_test, window_length=100, batch_size=32, device='cpu', layer_num=3, epochs = 10):
    decoder.eval()
    discriminator.eval()

    test_loss = 0
    correct = 0
    batch_cnt = 0

    recon_test_loss = 0
    recon_correct = 0 

    len_test = len(x_test)
    loss_function = nn.NLLLoss()

    layer_embedding, x_test, y_test = create_latent(x_test, y_test, layer_num, window_length, batch_size, epochs, device) # epochs could be changed

    with torch.no_grad():
        for i_batch in range(len(layer_embedding)):
            data = x_test[i_batch].to(device, dtype=torch.float)
            true_scores = y_test[i_batch].to(device)

            decode_raw_reshape = torch.reshape(data, (-1, 100, 9))
            discriminator_raw_output = discriminator(decode_raw_reshape)
            output = discriminator_raw_output[:, -1, :]

            # output = discriminator(data)
            test_loss += loss_function(output, true_scores).sum().item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability           

            correct += pred.eq(true_scores.view_as(pred)).sum().item()
            batch_cnt+=1

            embedding_gpu = layer_embedding[i_batch].to(device)
            recon_data = decoder(embedding_gpu)

            decode_output_reshape = torch.reshape(recon_data, (-1, 100, 9))
            # torch.save(decode_output_reshape, saved_data_path+str(layer_num)+"_layer_recon.pt")


            discriminator_output = discriminator(decode_output_reshape)
            recon_output = discriminator_output[:, -1, :]

            # recon_output = discriminator(recon_data)
            recon_test_loss += loss_function(recon_output, true_scores).sum().item()  # sum up batch loss
            recon_pred = recon_output.argmax(dim=1)  # get the index of the max log-probability
            recon_correct += recon_pred.eq(true_scores.view_as(recon_pred)).sum().item()


    test_loss /= (len_test-batch_size)
    recon_test_loss /= (len_test-batch_size)

    print('\n Raw data test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len_test, 100. * correct / (batch_cnt*batch_size)))
    print('\n Reconstructed data test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(recon_test_loss, recon_correct, len_test, 100. * recon_correct / (batch_cnt*batch_size)))  


if __name__== "__main__":

    epochs = 100
    axis_num = 9
    layer_num = 1
    batch_size = 32
    test_size = 0.3
    window_length = 100
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    x_train, x_test, y_train, y_test = split_dataset(test_size = test_size, window_length = window_length)

    decoder, discriminator = adversarial_train(x_train, y_train, layer_num=layer_num, window_length = window_length, batch_size = batch_size, epochs = epochs, device = device)

    model_test(decoder, discriminator, x_train, y_train, layer_num=layer_num, window_length = window_length, batch_size = batch_size, device = device, epochs = epochs)
