from torch.utils.data import Dataset
import torch
import numpy as np

class EEGEyeNetDataset(Dataset):
    def __init__(self, data_file,transpose = True):
        self.data_file = data_file
        print('loading data...')
        with np.load(self.data_file) as f: # Load the data array
            self.trainX = f['EEG']
            self.trainY = f['labels']
        print(self.trainY)
        if transpose:
            self.trainX = np.transpose(self.trainX, (0,2,1))[:,np.newaxis,:,:]

    def __getitem__(self, index):
        # Read a single sample of data from the data array
        X = torch.from_numpy(self.trainX[index]).float()
        y = torch.from_numpy(self.trainY[index,1:3]).float()
        # Return the tensor data
        return (X,y,index)

    def __len__(self):
        # Compute the number of samples in the data array
        return len(self.trainX)