import os
import pickle 
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

seed = 42
data_path = './harbox_sensys21_dataset/'

class IMUDataset(Dataset):
    def __init__(self, rootpath = data_path):
        super().__init__()
        self.X, self.y = process_dataset(window_length=50, scale = False) #50 Hz, so 1s.

    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx])
        sample = sample[None,:]
        return sample, self.y[idx]

def normalize(dataset, window_length):
    v = np.array(dataset)
    last_axis = v.shape[2]
    v = v.reshape(-1, last_axis)
    v_mean = np.mean(v, 0)
    v_std = np.std(v, 0)
    v = (v-v_mean)/v_std
    dataset =v.reshape(-1, window_length, last_axis)
    return dataset

def scale(v): #[-1, 1]
    # v:np.array [length, width, height]
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    x = 2*(v - v_min)/(v_max - v_min)-1
    return x

def load_data_with_id(id_list = ['1', '50', '100'], window_length = 100, axis_num = 9, data_path='./harbox_sensys21_dataset/'):
    dataset = {id_list[0]: {}, id_list[1]: {}, id_list[2]:{}}
    class_set = ['Call','Hop','typing','Walk','Wave']
    class_num = len(class_set)
    for user_id in list(dataset.keys()):
        for class_id in range(class_num):
            read_path = data_path+str(user_id) + '/' + str(class_set[class_id]) + '_train' + '.txt'
            if os.path.exists(read_path):
                tmp_original_data = read_data(read_path)
                tmp_coll = tmp_original_data.iloc[:, 1:axis_num+1].to_numpy().reshape(-1, window_length, axis_num)
                if class_id not in dataset[str(user_id)].keys():
                    dataset[str(user_id)][class_id]=[]
                dataset[str(user_id)][class_id].extend(tmp_coll)
    return dataset

def set_loader(config):
    imu_data = IMUDataset(rootpath = data_path)
    train_len = int(len(imu_data)*config.train_ratio)
    test_len = len(imu_data)-train_len
    train_set, test_set = torch.utils.data.random_split(imu_data, [train_len, test_len])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, num_workers=16, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, num_workers=16, pin_memory=True, drop_last=True)
    return train_loader, test_loader

def read_data(file_path):
    column_names = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
    data = pd.read_csv(file_path, sep=' ', header = None, names = column_names)
    return data

def load_data(window_length, axis_num=9, data_path=data_path):
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

def process_dataset(window_length, scale):
    if os.path.exists('harbox_dataset.pkl'):
        with open('harbox_dataset.pkl', 'rb') as f:
            x_dataset, y_dataset = pickle.load(f)
    else:
        dataset = load_data(window_length=window_length)
        x_dataset = []
        y_dataset = []
        for key in dataset.keys():
            x_dataset.extend(dataset[key])
            y_dataset.extend(key*np.ones(len(dataset[key])).astype(int))
        
        if scale == True:
            v = np.array(x_dataset)
            v_min = v.min(axis=(0, 1), keepdims=True)
            v_max = v.max(axis=(0, 1), keepdims=True)
            x_dataset = 2*(v - v_min)/(v_max - v_min)-1

        x_dataset = normalize(x_dataset, window_length)

        with open('harbox_dataset.pkl', 'wb') as f:
            dataset = (x_dataset, y_dataset)
            pickle.dump(dataset, f)
    return x_dataset, y_dataset


def split_dataset(train_size=0.8, window_length=50, scale = False):
    x_dataset, y_dataset = process_dataset(window_length = window_length, scale = scale)

    if train_size == 1:
        return np.array(x_dataset), np.array(y_dataset), [], []
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size = 1 - train_size, random_state=seed)
        x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
        return x_train, x_test, y_train, y_test