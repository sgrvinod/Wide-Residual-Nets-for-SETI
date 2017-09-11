from __future__ import print_function
import argparse
import os
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from wresnet_models import *
from h5_dataloaders import *
import pandas as pd
import ibmseti
import numpy as np
import ibmseti




def run(modelcheckpoint, normalizeData, simfile):
    """
    """

    model = wresnet34x2().cpu()

    if os.path.isfile(modelcheckpoint):
        print("=> Loading checkpoint '{}'".format(modelcheckpoint))
        checkpoint = torch.load(modelcheckpoint, map_location=lambda storage, loc: storage)
        best_acc = checkpoint['best_acc']
        print("This model had an accuracy of %.2f on the validation set." % (best_acc,))
        keys = checkpoint['state_dict'].keys()
        for old_key in keys:
            new_key = old_key.replace('module.', '')
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(old_key)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded checkpoint '{}' (epoch {})"
              .format(modelcheckpoint, checkpoint['epoch']))
    else:
        print("=> No model checkpoint found. Exiting")
        return None


    cudnn.benchmark = False

    # Load the Normalizer function
    h = h5py.File(normalizeData, 'r')
    mean = torch.FloatTensor(h['mean'][:])
    mean = mean.permute(2, 0, 1)
    std_dev = torch.FloatTensor(h['std_dev'][:])
    std_dev = std_dev.permute(2, 0, 1)
    h.close()
    normalize = transforms.Normalize(mean=mean,
                                     std=std_dev)

    # Load simulation data
    time_freq_resolution=(384, 512)
    aca = ibmseti.compamp.SimCompamp(open(simfile, 'rb').read())
    complex_data = aca.complex_data()
    complex_data = complex_data.reshape(time_freq_resolution[0], time_freq_resolution[1])
    complex_data = complex_data * np.hanning(complex_data.shape[1])
    cpfft = np.fft.fftshift(np.fft.fft(complex_data), 1)
    spectrogram = np.abs(cpfft)
    features = np.stack((np.log(spectrogram ** 2),
                         np.arctan(cpfft.imag / cpfft.real)), -1)


    # create FloatTensor, permute to proper dimensional order, and normalize
    data = torch.FloatTensor(features)
    data = data.permute(2, 0, 1)
    data = normalize(data)

    # The model expects a 4D tensor
    s = data.size()
    data = data.contiguous().view(1, s[0], s[1], s[2])
        
    input_var = torch.autograd.Variable(data, volatile=True)

    model.eval()

    softmax = torch.nn.Softmax()
    softmax.zero_grad()
    output = model(input_var)
    probs = softmax(output).data.view(7).tolist()
    
    return probs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SETI Classifier - Test Model')

    parser.add_argument('checkpoint', metavar='PATH',
                        help='path to model checkpoint')
    parser.add_argument('h5normalizedata', metavar='PATH',
                        help='path to hdf5 file with mean and std-dev tensors')
    parser.add_argument('singlefile', metavar='PATH',
                        help='path to SETI simulation file')

    args = parser.parse_args()

    probs = run(args.checkpoint, args.h5normalizedata, args.singlefile)

    print(probs)