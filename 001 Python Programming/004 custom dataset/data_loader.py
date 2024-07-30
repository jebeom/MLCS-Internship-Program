############################################
# Task : Data load
# Author: Jebeom Chae
# Date:   2024-06-30
# Reference : Hwanmoo Yong
###########################################
import numpy as np
import os, glob
import cv2

class DataLoader():

    def __init__(self, dataset_name = 'ball types', shuffle=False):
        self.path_to_baseball = os.path.join('./dataset', 'baseball', '*.jpeg')
        self.path_to_basketball = os.path.join('./dataset', 'basketball', '*.jpeg')

        self.baseball_path = glob.glob(self.path_to_baseball)
        self.basketball_path = glob.glob(self.path_to_basketball)

        self.baseball = np.empty((len(self.baseball_path), 225, 225))
        self.basketball = np.empty((len(self.basketball_path), 225, 225))

        self.baseball_label = np.zeros(len(self.baseball))    # baseball label = 0
        self.basketball_label = np.ones(len(self.basketball)) # basketball label = 1

        # Settings
        self.dataset_name = dataset_name
        self.shuffle = shuffle

    
    def create(self):

        for data_idx, data_path in enumerate(self.baseball_path):
            image = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
            self.baseball[data_idx] = cv2.resize(image, (225, 225), interpolation=cv2.INTER_AREA)
        
        for data_idx, data_path in enumerate(self.basketball_path):
            image = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
            self.basketball[data_idx] = cv2.resize(image, (225, 225), interpolation=cv2.INTER_AREA)

        data = np.concatenate((self.baseball, self.basketball), axis=0)
        label = np.concatenate((self.baseball_label, self.basketball_label), axis=0)


        if self.shuffle:
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data = data[indices]
            label = label[indices]

        train_size = int(len(data) * 0.7)
        valid_size = int(len(data) * 0.2)

        self.train, self.valid, self.test = np.split(data, [train_size, train_size + valid_size])
        self.train_label, self.valid_label, self.test_label = np.split(label, [train_size, train_size + valid_size])

        np.savez(os.path.join('./dataset', self.dataset_name+'_train.npz'), data = self.train, label = self.train_label)
        np.savez(os.path.join('./dataset', self.dataset_name+'_valid.npz'), data = self.valid, label = self.valid_label)
        np.savez(os.path.join('./dataset', self.dataset_name+'_test.npz'), data = self.test, label = self.test_label)

        
    def load(self):
        train_dataset = np.load(os.path.join('./dataset', self.dataset_name + '_train.npz'))
        valid_dataset = np.load(os.path.join('./dataset', self.dataset_name + '_valid.npz'))
        test_dataset = np.load(os.path.join('./dataset', self.dataset_name + '_test.npz'))
        
        self.train = train_dataset['data']
        self.valid = valid_dataset['data']
        self.test = test_dataset['data']
        self.train_label = train_dataset['label']
        self.valid_label = valid_dataset['label']
        self.test_label = test_dataset['label']

        return self.train, self.valid, self.test, self.train_label, self.valid_label, self.test_label

if __name__ == "__main__":
        dl = DataLoader()
        dl.create()
        dl.load()