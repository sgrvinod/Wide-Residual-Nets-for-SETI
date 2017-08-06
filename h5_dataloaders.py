import torch
from torch.utils.data import Dataset
import h5py


class h5Dataset(Dataset):
    """Custom Dataset object for loading train/val data directly from disk as required.

    Note: to be used in train.py only!

    The data is input tensor ([log(amplitude^2) and phase] x time steps x frequency bins) and target (true label index).

    Args:
        h5_filepath (path): path to the hdf5 file that contains the train/val data.
        folds (list of ints): indices of folds to include as train/val data.
        target_class_index_mapping: mapping of class name to class index
        transform (torchvision.transforms object): transformations to perform on the input data
        target_transform (torchvision.transforms object): transformations to perform on the target data
    """

    def __init__(self, h5_filepath, folds, target_class_index_mapping, transform=None, target_transform=None):
        self.folds = folds
        self.h5_filepath = h5_filepath
        self.target_class_index_mapping = target_class_index_mapping
        self.transform = transform
        self.target_transform = target_transform
        self.h = h5py.File(self.h5_filepath, 'r')
        signal_index_mapping = {}
        n_signals = 0
        for f in folds:
            assert 'fold' + str(f) + '_data' in self.h.keys() and 'fold' + str(
                f) + '_target' in self.h.keys(), 'No fold %d found in hdf5 file!'
            for i, signal in enumerate(range(self.h['fold' + str(f) + '_data'].shape[0] - 1)):
                signal_index_mapping[i + n_signals] = (f, i)
            n_signals = len(signal_index_mapping.keys())
        self.signal_index_mapping = signal_index_mapping
        self.n_signals = n_signals

    def __getitem__(self, i):
        f, f_i = self.signal_index_mapping[i]
        data = torch.FloatTensor(self.h['fold' + str(f) + '_data'][f_i])
        data = data.permute(2, 0, 1)
        target = self.target_class_index_mapping[self.h['fold' + str(f) + '_target'][f_i][0]]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return self.n_signals


class h5TestDataset(Dataset):
    """Custom Dataset object for loading test data directly from disk as required.

    Note: to be used in test.py only!

    The data is input tensor ([log(amplitude^2) and phase] x time steps x frequency bins) and UUID (index of UUID).

    Args:
        h5_filepath (path): path to the hdf5 file that contains the train/val data.
        transform (torchvision.transforms object): transformations to perform on the input data
    """

    def __init__(self, h5_filepath, transform=None):
        self.h5_filepath = h5_filepath
        self.transform = transform
        self.h = h5py.File(self.h5_filepath, 'r')
        self.n_signals = self.h['data'][:].shape[0]

    def __getitem__(self, i):
        data = torch.FloatTensor(self.h['data'][i])
        data = data.permute(2, 0, 1)
        if self.transform is not None:
            data = self.transform(data)
        return data, i

    def __len__(self):
        return self.n_signals
