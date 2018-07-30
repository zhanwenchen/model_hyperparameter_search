import h5py
import numpy as np
import os
from torch.utils.data import Dataset
import torch

def loader():
    # Train data
    fname_train_images = 'data/train-images-idx3-ubyte'  # the training set image file path
    fname_train_labels = 'data/train-labels-idx1-ubyte'  # the training set label file path
    fname_test_images = 'data/t10k-images-idx3-ubyte'  # the training set label file path
    fname_test_labels = 'data/t10k-labels-idx1-ubyte'  # the training set label file path

    # open the label file and load it to the "train_labels"
    with open(fname_train_labels, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        train_labels = np.fromfile(flbl, dtype=np.uint8)

    # open the label file and load it to the "test_labels"
    with open(fname_test_labels, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        test_labels = np.fromfile(flbl, dtype=np.uint8)

    # open the image file and load it to the "train_images"
    with open(fname_train_images, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        train_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(train_labels), rows, cols)


    # open the image file and load it to the "test_images"
    with open(fname_test_images, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        test_images = np.fromfile(fimg, dtype=np.uint8).reshape(len(test_labels), rows, cols)


class MnistDataset(Dataset):
    def __init__(self, fname):
        """
        Args:
            fname: file name for aperture domain data
            num_samples: number of samples to use from data set
            target_is_data: return data as the target (autoencoder)
        """
        self.fname = fname
        self.num_samples = num_samples

        # check if files exist
        if not os.path.isfile(fname):
            raise IOError('dataloader.py: ' + fname + ' does not exist.')

        # Open file
        f = h5py.File(fname, 'r')

        # Get number of samples available for each type
        real_available = f['/' + str(k) + '/X/real'].shape[0]
        imag_available = f['/' + str(k) + '/X/imag'].shape[0]

        # load the data
        inputs = np.hstack([f['/' + str(k) + '/X/real'][0:self.num_samples],
                            f['/' + str(k) + '/X/imag'][0:self.num_samples]])
        if target_is_data:
            targets = np.hstack([f['/' + str(k) + '/X/real'][0:self.num_samples],
                                 f['/' + str(k) + '/X/imag'][0:self.num_samples]])
        else:
            targets = np.hstack([f['/' + str(k) + '/Y/real'][0:self.num_samples],
                                 f['/' + str(k) + '/Y/imag'][0:self.num_samples]])

        # convert data to single precision pytorch tensors
        self.data_tensor = torch.from_numpy(inputs).float()
        self.target_tensor = torch.from_numpy(targets).float()

        # close file
        f.close()

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]


# @author PyTorch/Vision
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


# @author PyTorch/Vision
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()
