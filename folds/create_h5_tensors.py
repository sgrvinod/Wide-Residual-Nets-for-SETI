import ibmseti
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import numpy as np
import h5py
from tqdm import tqdm


def create_tensors_hdf5_logmod2_ph(dataset_name, dataset_folder, output_folder, number_of_folds,
                                   time_freq_resolution):
    """
    Reads signals, divides into stratified folds, performs windowing and FFT, generates two features:
    log(amplitude^2) and phase of signal.

    Note: to be used for training/val, in h5Dataset object (from h5_dataloaders.py) only!

    Writes the folds' tensors, overall mean and standard deviation tensors, to a specified hdf5 file.

    Args:
        dataset_name (string): name of signal dataset.
        dataset_folder (path): folder containing signal files.
        output_folder (path): output location of hdf5 file.
        number_of_folds (int): number of stratified folds to divide the signals into.
        time_freq_resolution (tuple of ints): number of time steps and frequency windows.
    """
    features_name = 'logmod2-ph'
    number_of_features = 2

    # Check for some things
    assert time_freq_resolution[0] * time_freq_resolution[1] == 32 * 6144 and len(
        time_freq_resolution) == 2, 'Invalid time-frequency resolution!'
    assert os.path.isdir(dataset_folder), 'Invalid dataset directory!'

    # Read CSV
    files_in_dataset_folder = os.listdir(dataset_folder)
    signal_classification = pd.read_csv(
        os.path.join(dataset_folder, [f for f in files_in_dataset_folder if f.endswith('.csv')][0]))
    assert 'UUID' in signal_classification.columns and 'SIGNAL_CLASSIFICATION' in signal_classification.columns
    signal_classification = shuffle(signal_classification)
    signal_classification = signal_classification.reset_index(drop=True)

    # Check distribution of classes
    print "Classes' count for all folds:"
    print signal_classification.groupby(['SIGNAL_CLASSIFICATION']).count()

    # Create folds
    uuids = signal_classification['UUID'].to_frame()
    classes = signal_classification['SIGNAL_CLASSIFICATION']
    skf = StratifiedKFold(n_splits=5).split(uuids, classes)
    fold_signals = []
    for i, fold in enumerate(skf):
        dataframe_slice = signal_classification.iloc[fold[1], :]
        print "Classes' count for Fold %d:" % (i + 1)
        print dataframe_slice.groupby(['SIGNAL_CLASSIFICATION']).count()
        fold_signals.append(zip(list(dataframe_slice['UUID']), list(dataframe_slice['SIGNAL_CLASSIFICATION'])))

    # Create/open hdf5 file
    hdf5_file_name = dataset_name + '__' + str(number_of_folds) + 'folds__' + str(
        time_freq_resolution[0]) + 't__' + str(
        time_freq_resolution[1]) + 'f__' + features_name + '.hdf5'

    # Create tensors and write to file
    print "\nWriting %d folds to %s..." % (number_of_folds, hdf5_file_name)
    for i, fold in enumerate(fold_signals):
        with h5py.File(os.path.join(output_folder, hdf5_file_name), 'a') as h:
            fold_data = h.create_dataset('fold' + str(i + 1) + '_data',
                                         (1, time_freq_resolution[0], time_freq_resolution[1], number_of_features),
                                         maxshape=(
                                             None, time_freq_resolution[0], time_freq_resolution[1],
                                             number_of_features))
            fold_target = h.create_dataset('fold' + str(i + 1) + '_target', (1, 1), dtype='|S13', maxshape=(None, 1))

            print "\nPopulating data and targets for Fold %d... " % (i + 1)
            for j, signal in enumerate(tqdm(fold)):
                try:
                    with open(os.path.join(dataset_folder, signal[0] + '.dat'), 'r') as f:
                        aca = ibmseti.compamp.SimCompamp(f.read())
                except IOError:
                    continue
                complex_data = aca.complex_data()
                complex_data = complex_data.reshape(time_freq_resolution[0], time_freq_resolution[1])
                complex_data = complex_data * np.hanning(complex_data.shape[1])
                cpfft = np.fft.fftshift(np.fft.fft(complex_data), 1)
                spectrogram = np.abs(cpfft)
                features = np.stack((np.log(spectrogram ** 2),
                                     np.arctan(cpfft.imag / cpfft.real)), -1)
                fold_data[fold_data.shape[0] - 1] = features
                fold_target[fold_data.shape[0] - 1] = np.array([signal[1]])
                if j == len(fold) - 1:
                    break  # Don't resize/add an extra row if after the last signal in the fold
                fold_data.resize(fold_data.shape[0] + 1, axis=0)
                fold_target.resize(fold_target.shape[0] + 1, axis=0)
                del features
        del fold_data, fold_target
    print "\nFolds written to %s" % hdf5_file_name

    # Calculate mean tensor (for normalization, later)
    print "\nCalculating mean tensor (across frequency bins and channels) without loading folds into memory..."
    total_number_of_signals = 0
    for i in range(number_of_folds):
        print "\nReviewing Fold %d..." % (i + 1)
        with h5py.File(os.path.join(output_folder, hdf5_file_name), 'r') as h:
            dset = h['fold' + str(i + 1) + '_data']
            if i == 0:
                sum_tensor = np.zeros(dset.shape[1:], dtype=float)
            for j in tqdm(range(dset.shape[0])):
                sum_tensor = sum_tensor + dset[j]
            total_number_of_signals = total_number_of_signals + dset.shape[0]
        del dset
    mean_tensor = sum_tensor / total_number_of_signals
    mean_tensor = np.mean(mean_tensor, axis=0)
    mean_tensor = np.repeat(mean_tensor[np.newaxis, :, :], time_freq_resolution[0], axis=0)
    print "\nCalculated mean tensor (across frequency bins and channels)."

    # Calculate standard deviation tensor (for normalization, later)
    print "\nCalculating std-deviation tensor (across frequency bins and channels) without loading folds into memory..."
    total_number_of_signals = 0
    for i in range(number_of_folds):
        print "\nReviewing Fold %d..." % (i + 1)
        with h5py.File(os.path.join(output_folder, hdf5_file_name), 'r') as h:
            dset = h['fold' + str(i + 1) + '_data']
            if i == 0:
                sum_of_squared_differences_tensor = np.zeros(dset.shape[1:], dtype=float)
            for j in tqdm(range(dset.shape[0])):
                assert mean_tensor.shape == dset[j].shape
                sum_of_squared_differences_tensor = sum_of_squared_differences_tensor + (dset[j] - mean_tensor) ** 2
            total_number_of_signals = total_number_of_signals + dset.shape[0]
        del dset
    mean_of_squared_differences_tensor = sum_of_squared_differences_tensor / total_number_of_signals
    mean_of_squared_differences_tensor = np.mean(mean_of_squared_differences_tensor, axis=0)
    std_deviation_tensor = np.sqrt(mean_of_squared_differences_tensor)
    std_deviation_tensor = np.repeat(std_deviation_tensor[np.newaxis, :, :], time_freq_resolution[0], axis=0)
    print "\nCalculated std-deviation tensor (across frequency bins and channels)."

    assert mean_tensor.shape == std_deviation_tensor.shape

    # Store mean and standard deviation tensors to the hdf5 file
    print "\nStoring these to the same hdf5 file..."
    with h5py.File(os.path.join(output_folder, hdf5_file_name), 'a') as h:
        mean = h.create_dataset('mean', mean_tensor.shape)
        std_dev = h.create_dataset('std_dev', std_deviation_tensor.shape)
        mean[:] = mean_tensor
        std_dev[:] = std_deviation_tensor

    print "\nAll done!"


def create_test_tensors_hdf5_logmod2_ph(dataset_name, dataset_folder, output_folder, time_freq_resolution):
    """
    Reads signals, performs windowing and FFT, generates two features:
    log(amplitude^2) and phase of signal.

    Note: to be used for testing, in h5TestDataset object (from h5_dataloaders.py) only!

    Writes the test data tensor to a specified hdf5 file.

    Args:
        dataset_name (string): name of signal dataset.
        dataset_folder (path): folder containing signal files.
        output_folder (path): output location of hdf5 file.
        time_freq_resolution (tuple of ints): number of time steps and frequency windows.
    """
    features_name = 'logmod2-ph'
    number_of_features = 2

    # Check for some things
    assert time_freq_resolution[0] * time_freq_resolution[1] == 32 * 6144 and len(
        time_freq_resolution) == 2, 'Invalid time-frequency resolution!'
    assert os.path.isdir(dataset_folder), 'Invalid dataset directory!'

    # Read CSV and get UUIDs
    files_in_dataset_folder = os.listdir(dataset_folder)
    signal_classification = pd.read_csv(
        os.path.join(dataset_folder, [f for f in files_in_dataset_folder if f.endswith('.csv')][0]))
    assert 'UUID' in signal_classification.columns
    uuids = list(signal_classification['UUID'])
    print "There are %d signals in this test set." % len(uuids)

    # HDF5 file name
    hdf5_file_name = 'TEST__' + dataset_name + '__' + str(
        time_freq_resolution[0]) + 't__' + str(
        time_freq_resolution[1]) + 'f__' + features_name + '.hdf5'

    # Create tensors and write to file
    print "\nWriting tensors to %s..." % (hdf5_file_name)
    with h5py.File(os.path.join(output_folder, hdf5_file_name), 'a') as h:
        h_data = h.create_dataset('data',
                                  (len(uuids), time_freq_resolution[0], time_freq_resolution[1], number_of_features))
        h_uuids = h.create_dataset('uuids', shape=(len(uuids), 1), dtype='|S50')
        for j, uuid in enumerate(tqdm(uuids)):
            try:
                with open(os.path.join(dataset_folder, uuid + '.dat'), 'r') as f:
                    aca = ibmseti.compamp.SimCompamp(f.read())
            except IOError:
                continue
            complex_data = aca.complex_data()
            complex_data = complex_data.reshape(time_freq_resolution[0], time_freq_resolution[1])
            complex_data = complex_data * np.hanning(complex_data.shape[1])
            cpfft = np.fft.fftshift(np.fft.fft(complex_data), 1)
            spectrogram = np.abs(cpfft)
            features = np.stack((np.log(spectrogram ** 2),
                                 np.arctan(cpfft.imag / cpfft.real)), -1)
            h_data[j] = features
            h_uuids[j] = np.array([uuid])
            del features
    del h_data, h_uuids
    print "\nTest data and UUIDs written to %s" % hdf5_file_name
    print "\nVerifying that things are in the same order as in the CSV..."
    with h5py.File(os.path.join(output_folder, hdf5_file_name), 'r') as h:
        for j, uuid in enumerate(tqdm(uuids)):
            if uuid != h['uuids'][:][j][0]:
                print uuid, h['uuids'][:][j][0]
                raise ValueError("Value at index %d differs - %s != %s!" % (j + 1, uuid, h['uuids'][:][j][0]))

    print "All done!"


if __name__ == '__main__':
    # create_tensors_hdf5_logmod2_ph(dataset_name='primary_full_v3',
    #                                dataset_folder='../sti raw files/primary_full_dataset_v3',
    #                                output_folder='./',
    #                                number_of_folds=5,
    #                                time_freq_resolution=(32 * 12, 6144 / 12))

    create_test_tensors_hdf5_logmod2_ph(dataset_name='testset_final',
                                        dataset_folder='../sti raw files/primary_testset_final_v3',
                                        output_folder='./',
                                        time_freq_resolution=(32 * 12, 6144 / 12))
