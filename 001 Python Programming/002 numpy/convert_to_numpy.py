import os
import glob
import numpy as np
import random

# Process 1: Read the converted 2d dataset files in 'outputs' directory
path_to_train_2d_datasets = os.path.join('..', '001 csv', 'outputs', 'train', '*.csv')
train_2d_files = glob.glob(path_to_train_2d_datasets)

path_to_test_2d_datasets = os.path.join('..', '001 csv', 'outputs', 'test', '*.csv')
test_2d_files = glob.glob(path_to_test_2d_datasets)

# Process 2: Create empty 3D numpy arrays to save loaded 2D MNIST data
train = np.empty((len(train_2d_files), 28, 28))
train_label = np.empty((len(train_2d_files)))

test = np.empty((len(test_2d_files), 28, 28))
test_label = np.empty((len(test_2d_files)))

# Process 3: Load data into the numpy arrays
for data_idx, data_path in enumerate(train_2d_files):
    # ex: '../001 csv/outputs/train/#0-#1.csv' -> '0'
    filename = os.path.basename(data_path)
    train_label[data_idx] = int(filename.split('-')[0].replace('#', ''))
    
    csv_data = np.loadtxt(data_path, delimiter=',') 
    for i in range(28):
        for j in range(28):
            train[data_idx, i, j] = csv_data[i, j]

for data_idx, data_path in enumerate(test_2d_files):
    # ex: '../001 csv/outputs/test/#0-#1.csv' -> '0'
    filename = os.path.basename(data_path)
    test_label[data_idx] = int(filename.split('-')[0].replace('#', ''))
    
    csv_data = np.loadtxt(data_path, delimiter=',') 
    for i in range(28):
        for j in range(28):
            test[data_idx, i, j] = csv_data[i, j]

# Process 4: Concat the train and test arrays into one array, so that we can split into three different arrays. (train, valid, test)
data = np.concatenate((train, test), axis=0)
label = np.concatenate((train_label, test_label), axis=0)

# Process 5: Using the random library, shuffle the numpy array
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]

# Process 6: Split the numpy array into train, validation, and test arrays. Each array has a ratio of 7:2:1 respectively.
train_size = int(len(data) * 0.7)
valid_size = int(len(data) * 0.2)

train, valid, test = np.split(data, [train_size, train_size + valid_size])
train_label, valid_label, test_label = np.split(label, [train_size, train_size + valid_size])

# Process 7: Save the splitted arrays into .npz files.
np.savez("train_x.npz", train_x=train)
np.savez("train_y.npz", train_y=train_label)

np.savez("valid_x.npz", valid_x=valid)
np.savez("valid_y.npz", valid_y=valid_label)

np.savez("test_x.npz", test_x=test)
np.savez("test_y.npz", test_y=test_label)
