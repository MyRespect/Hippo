import inspect
import os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d, uniform_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,2.25)
fig, axes = plt.subplots(1, 4)

font = {'family' : 'arial',
        'weight'   : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

seed = 42
n_classes = 4
data_path = './harbox_sensys21_dataset/'
saved_data_path = './saved_recon_files/'
class_set_0 = ['Call','Hop','Walk','Wave']

random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator_fix = torch.Generator(device=device)
generator_fix.manual_seed(2147483647)

def read_data(file_path):
    column_names = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
    data = pd.read_csv(file_path, sep=' ', header = None, names = column_names)
    return data

def load_data(window_length=100, axis_num=9, data_path='./harbox_sensys21_dataset/'):
    dataset = {}
    class_set = ['Call','Hop', 'Walk','Wave']
    class_num = len(class_set)
    for user_id in range(1, 121):
        for class_id in range(class_num):
            read_path = data_path +  str(user_id) + '/' + str(class_set[class_id]) + '_train' + '.txt'  # "_train" is from the train, we have another set for test
            if os.path.exists(read_path):
                tmp_original_data = read_data(read_path)
                tmp_coll = tmp_original_data.iloc[:, 1:axis_num+1].to_numpy().reshape(-1, window_length, axis_num)
                if class_id not in dataset.keys():
                    dataset[class_id]=[]
                dataset[class_id].extend(tmp_coll)
    return dataset

def load_data_with_id(window_length = 100, axis_num = 9, data_path='../harbox_sensys21_dataset/'):
    dataset = {'1': {}, '50': {}}
    class_set = ['Call','Hop', 'Walk','Wave']
    class_num = len(class_set)
    for user_id in [1, 50]:
        for class_id in range(class_num):
            read_path = data_path+str(user_id) + '/' + str(class_set[class_id]) + '_train' + '.txt'
            if os.path.exists(read_path):
                tmp_original_data = read_data(read_path)
                tmp_coll = tmp_original_data.iloc[:window_length, 1:axis_num+1].to_numpy()
                tmp_coll = tmp_coll.reshape(-1, window_length, axis_num)
                if class_id not in dataset[str(user_id)].keys():
                    dataset[str(user_id)][class_id]=[]
                dataset[str(user_id)][class_id].extend(tmp_coll)
    return dataset

def normalize(v):
    # v:np.array [length, width, height]
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    x = 2*(v - v_min)/(v_max - v_min)-1
    return x

# if __name__ == "__main__":
#     window_length = 2000

#     dataset = load_data_with_id(window_length = window_length)
#     x = np.arange(0, 90)

#     color = ['r', 'g', 'b', 'c', 'm']
#     for user_id in [1]:
#         for class_id in range(n_classes):
#             axes[class_id].plot(x, np.mean(dataset[str(user_id)][class_id][0][:,0:3], axis=1)[5:95], '.-', color=color[class_id], linewidth=2.0)
#             axes[class_id].set_title(class_set_0[class_id], fontdict=font)
#             axes[class_id].set_yticks([])
#             axes[class_id].set_xticks([])
#             if class_id == 0:
#                 axes[class_id].set_ylabel('Raw Data', fontdict=font)
#             # axes[class_id].plot(x, np.mean(dataset[str(user_id)][class_id][0][:,3:6], axis=1), '.-', linewidth=2.0)
#     fig = plt.gcf()
#     plt.tight_layout()
#     plt.show()
#     fig.savefig('time_series.pdf', dpi=300, transparent=True, bbox_inches='tight')

    # from scipy import signal

    # for user_id in [1]:
    #     for class_id in range(n_classes):
    #         tmp = np.mean(dataset[str(user_id)][class_id][0][:window_length,0:1], axis=1)
    #         powerSpectrum, freqenciesFound, time, imageAxis = axes[class_id].specgram(tmp, Fs=50, cmap="bwr", NFFT=64, noverlap = 16, window=np.hamming(64))
    #         # f, t, Sxx = signal.spectrogram(np.mean(dataset[str(user_id)][class_id][0][:,0:1], axis=1), fs=50, nperseg=50, scaling = 'spectrum')
    #         # axes[class_id].pcolormesh(t, f, Sxx, shading='gouraud')
    #         axes[class_id].set_title(class_set_0[class_id], fontdict=font)
    #         axes[class_id].set_xticks([])
    #         if class_id == 0:
    #             axes[class_id].tick_params(axis='y', which='major', labelsize=24)
    #         else:
    #             axes[class_id].set_yticks([])
    #         # axes[class_id].plot(x, np.mean(dataset[str(user_id)][class_id][0][:,3:6], axis=1), '.-', linewidth=2.0)
    #         if class_id == 0:
    #             axes[class_id].set_ylabel('Raw Data', fontdict=font)
    # fig = plt.gcf()
    # plt.tight_layout()
    # plt.show()
    # fig.savefig('specgram.pdf', dpi=300, transparent=True, bbox_inches='tight')

if __name__ == "__main__":

    dataset = np.load('data_3.npy')*10
    draw_timeseries = False
    draw_spectrogram = True


#     if draw_timeseries == True:
#         dataset = dataset[:,10,:,:]  #l3:9, l1-l2:10,
#         print(np.shape(dataset))
#         window_length = 100
#         x = np.arange(0, 90)
#         color = ['r', 'g', 'b', 'c', 'm']
#         for user_id in [1]:
#             for class_id in range(n_classes):
#                 # tmp = np.mean(dataset[str(user_id)][class_id][0][:,0:3], axis=1)
#                 if class_id >=2:
#                     class_idd=class_id+1
#                 else:
#                     class_idd=class_id
#                 tmp = np.mean(dataset[class_idd].reshape(window_length, 9)[:,0:3], axis=1)
#                 tmp = uniform_filter(tmp, size=3)[5:95]
#                 axes[class_id].plot(x, tmp, '.-', color=color[class_id], linewidth=2.0)
#                 # axes[class_id].set_title(class_set_0[class_id])
#                 axes[class_id].set_yticks([])
#                 axes[class_id].set_xticks([])
#                 # axes[class_id].tick_params(axis='both', which='major', labelsize=20)
#                 if class_id == 0:
#                     axes[class_id].set_ylabel('Granularity-1', fontdict=font)
#                 # axes[class_id].plot(x, np.mean(dataset[str(user_id)][class_id][0][:,3:6], axis=1), '.-', linewidth=2.0)
#         fig = plt.gcf()
#         plt.tight_layout()
#         plt.show()
#         fig.savefig('time_series_l1.pdf', dpi=300, transparent=True, bbox_inches='tight')

    if draw_spectrogram == True:
        window_length = 2000
        x = np.arange(0, window_length)
        for user_id in [1]:
            for class_id in range(n_classes):
                if class_id >=2:
                    class_idd=class_id+1
                else:
                    class_idd=class_id
                # tmp = np.mean(dataset[str(user_id)][class_id][0][:window_length,0:1], axis=1)
                tmp = dataset[class_idd].reshape(window_length, 9)
                tmp = tmp[:,0]
                powerSpectrum, freqenciesFound, time, imageAxis = axes[class_id].specgram(tmp, Fs=50, cmap="bwr", NFFT=64, noverlap = 16, window=np.hamming(64))
                # axes[class_id].set_title(class_set_0[class_id])
                if class_id == 0:
                    axes[class_id].tick_params(axis='y', which='major', labelsize=24)
                else:
                    axes[class_id].set_yticks([])
                # axes[class_id].set_xticks([])
                axes[class_id].tick_params(axis='x', which='major', labelsize=24)
                # axes[class_id].plot(x, np.mean(dataset[str(user_id)][class_id][0][:,3:6], axis=1), '.-', linewidth=2.0)
                if class_id == 0:
                    axes[class_id].set_ylabel('Granularity-3', fontdict=font)
        fig = plt.gcf()
        plt.tight_layout()
        plt.show()
        fig.savefig('specgram_layer_l3.pdf', dpi=300, transparent=True, bbox_inches='tight')