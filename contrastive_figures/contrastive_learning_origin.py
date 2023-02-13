import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os, random
import numpy as np
import pandas as pd
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

random.seed(0)

window_length=100

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
    
    v = np.array(x_dataset)
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    x_dataset = 2*(v - v_min)/(v_max - v_min)-1 # [34115, 100, 9]

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size = test_size, random_state=42)
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    return x_train, x_test, y_train, y_test


def contrastive_loss(logits_embedding: Tensor, labels: Tensor):
    loss = 0
    len_embedding = len(logits_embedding)

    # print(len_embedding)

    dis_matrix = torch.cdist(logits_embedding, logits_embedding)

    # print(dis_matrix.size())

    for idx1 in range(len_embedding):
        for idx2 in range(len_embedding):
            if labels[idx1] == labels[idx2]:
                y = 0
            else:
                y = 1
            dd = dis_matrix[idx1][idx2]
            # print(dd)
            loss+=(1-y)*dd*dd+y*(80-dd)*(80-dd) #30
    loss /=(len_embedding*len_embedding)

    return loss

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
        return x

def cnn_train(model, loss_function, optimizer, x_train, y_train, window_length=100, batch_size=256, epochs=1, device='cpu'):
    model.train()
    for epoch in range(epochs):
        correct = 0
        batch_cnt=0
        for idx in range(0, len(x_train)-batch_size, batch_size):
            sample = torch.tensor(x_train[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            true_scores = torch.tensor(np.array([y_train[idx:idx+batch_size]]), dtype=torch.long).reshape(batch_size)
            sample = sample.to(device)
            true_scores = true_scores.to(device)
            model.zero_grad()
            x_embedding = model(sample)
            loss = contrastive_loss(x_embedding, true_scores)
            loss.backward()
            optimizer.step()
            batch_cnt+=1
        clear_output(wait=True)
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
    return model

def cnn_test(model, loss_function, x_test, y_test, window_length=100, batch_size=32, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    len_test = len(x_test)
    save_embed_label = []
    with torch.no_grad():
        for idx in range(0, len_test-batch_size, batch_size):
            data=torch.tensor(x_test[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            true_scores = torch.tensor(np.array([y_test[idx:idx+batch_size]]), dtype=torch.long).reshape(batch_size)
            data = data.to(device)
            true_scores = true_scores.to(device)
            output = model(data)
            save_embed_label.append([output, true_scores])
    torch.save(save_embed_label, "save_embed_label_raw.pt")


if __name__ == "__main__":

    window_length=100
    axis_num=9
    batch_size=32
    epochs=30

    x_train, x_test, y_train, y_test = split_dataset(test_size = 0.5, window_length=window_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(window_length=window_length, axis_num=axis_num, label_num=5).to(device)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=False)

    # trained_model = cnn_train(model, loss_function, optimizer, x_train, y_train, window_length, batch_size, epochs, device)
    # torch.save(trained_model, 'contrastive_harbox_model.pt')

    trained_model = torch.load('contrastive_harbox_model.pt')
    print("Finish training and start testing: ")
    cnn_test(trained_model, loss_function, x_train, y_train, batch_size=batch_size, device=device)