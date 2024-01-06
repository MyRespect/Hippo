import os 
import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

# Download the dataset from blow:
# https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

def read_file(file_path, window_size=32):
    step_size = int(window_size/2)
    signal = np.loadtxt(file_path, delimiter=',').tolist()
    result = []
    for i in range(0, len(signal)-window_size*2+1, step_size): # increase number of samples
        result.append(signal[i:i + window_size*2])
    return result

def read_person(folder_path):
    person_list = []
    for file in os.listdir(folder_path):
        combined_path = os.path.join(folder_path, file)
        signal = read_file(combined_path)
        person_list.append(signal)
    return person_list

def read_activity(folder_path, person_list):
    activity_list = []
    for folder in os.listdir(folder_path): # p1-p8
        if folder in person_list:
            continue
        combined_path = os.path.join(folder_path, folder)
        signal = read_person(combined_path)
        activity_list.append(signal)
    return activity_list

def load_dataset_dict(folder_path, person_list):
    dataset = {}
    for folder in os.listdir(folder_path):
        combined_path = os.path.join(folder_path, folder)
        activity_list = read_activity(combined_path, person_list)
        # print(np.shape(activity_list)) # (8, 60, 4, 50, 45)
        dataset[folder] = activity_list
    return dataset

class daily_sports_dataset(Dataset):
    def __init__(self, folder_path = './daily_sports_activities/', saved = False, local_data_name="daily_sports_dataset", excluded_person_list=[]):
        self.X = []
        self.y = []

        if saved == False:
            dataset_dict = load_dataset_dict(folder_path, excluded_person_list)
            for idx, key in enumerate(dataset_dict.keys()):
                shape_tuple = np.shape(dataset_dict[key]) # (8, 60, 4, 64, 45) # window_size=32
                self.y.extend([int(idx)] * shape_tuple[0] * shape_tuple[1] * shape_tuple[2])

                shaped_X = np.reshape(dataset_dict[key], (-1, 1, shape_tuple[3], shape_tuple[4])).astype(np.float32)
                shaped_X = np.pad(shaped_X, ((0,0), (0,0), (0, 0), (9, 10)), 'reflect') # (-1, 1, 64, 64)
                self.X.extend(shaped_X)

                dataset_tuple = (self.X, self.y)

                torch.save(dataset_tuple, local_data_name+str(len(excluded_person_list))+'.pt')
        else:
            dataset_tuple = torch.load(local_data_name+str(len(excluded_person_list))+'.pt')
            self.X, self.y = dataset_tuple
        print("Loaded Dataest: ", np.shape(self.X), np.shape(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx])
        return sample, self.y[idx]

if __name__ == "__main__":
    DailySportsDataset = daily_sports_dataset() # daily_sports_dataset0.pt includes all person
    # DailySportsDataset1 = daily_sports_dataset(excluded_person_list=['p7', 'p8']) # daily_sports_dataset2.pt
    # DailySportsDataset2 = daily_sports_dataset(excluded_person_list=['p1', 'p2', 'p3', 'p4', 'p5', 'p6']) # daily_sports_dataset6.pt