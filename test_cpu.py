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

parser = argparse.ArgumentParser(description='SETI Classifier - Test Model')

parser.add_argument('arch', metavar='PATH',
                    help='architecture to use')
parser.add_argument('checkpoint', metavar='PATH',
                    help='path to model checkpoint')
parser.add_argument('h5data', metavar='PATH',
                    help='path to hdf5 file with test data')
parser.add_argument('h5normalizedata', metavar='PATH',
                    help='path to hdf5 file with mean and std-dev tensors')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

# Available models
# model_archs = ['resnet18', 'resnet34', 'resnet50', 'resnet86', 'resnet101', 'resnet131', 'resnet203', 'resnet152',
#                'resrnn2x2', 'resrnn2x3', 'resrnn3x2', 'resrnn3x3', 'resrnn3x10', 'wresnet28x10', 'wresnet16x8',
#                'wresnet34x2', 'wresnet40x10', 'wresnet28x20', 'densenet161', 'densenet201', 'dpn92', 'dpn98',
#                'dpn131']
model_archs = ['wresnet34x2']


def main():
    """
    Load model's graph, loss function, optimizer, dataloaders.

    Perform testing.
    """

    global args
    args = parser.parse_args()

    print("\n\nChosen args:")
    print(args)

    assert args.arch in model_archs
    model = eval(args.arch + '()').cpu()


    if os.path.isfile(args.checkpoint):
        print("=> Loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print("This model had an accuracy of %.2f on the validation set." % (best_acc,))
        keys = checkpoint['state_dict'].keys()
        for old_key in keys:
            new_key = old_key.replace('module.', '')
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(old_key)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded checkpoint '{}' (epoch {})"
              .format(args.checkpoint, checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(args.checkpoint))


    cudnn.benchmark = False

    # Store {index->UUID} mapping in the order in the test set, to keep track of the UUIDs of the data in the DataLoader
    # This isn't really required since the DataLoader returns in the original order with shuffle=False, but hey...
    print('UUID mapping... ')
    h = h5py.File(args.h5data, 'r')
    global uuid_index_mapping
    uuid_index_mapping = {}
    for i in range(h['uuids'][:].shape[0]):
        uuid_index_mapping[i] = h['uuids'][:][i][0]
    h.close()

    # Normalizer
    print('Normalizing signals...')
    h = h5py.File(args.h5normalizedata, 'r')
    mean = torch.FloatTensor(h['mean'][:])
    mean = mean.permute(2, 0, 1)
    std_dev = torch.FloatTensor(h['std_dev'][:])
    std_dev = std_dev.permute(2, 0, 1)
    h.close()
    normalize = transforms.Normalize(mean=mean,
                                     std=std_dev)

    # Custom dataloader
    print('Instantiating test loader')
    test_loader = torch.utils.data.DataLoader(
        h5TestDataset(args.h5data, transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    test(test_loader, model)


def test(test_loader, model):
    """
    Perform testing.
    """

    print('Perform testing')

    model.eval()  # eval mode

    all_probs = []
    all_uuids = []

    batch_time = AverageMeter()  # forward prop. time this batch

    start = time.time()

    softmax = torch.nn.Softmax()  # need this, since there is no longer a loss layer

    for i, (input, uuids) in enumerate(test_loader):

        softmax.zero_grad()

        # Store UUIDs associated with this batch, in the right order
        uuids = list(uuids.numpy().ravel())
        all_uuids.extend(uuids)

        input_var = torch.autograd.Variable(input, volatile=True).cpu()

        output = model(input_var)
        probs = softmax(output)
        
        all_probs.append(probs.data)

        batch_time.update(time.time() - start)
        start = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(test_loader),
                                                                                    batch_time=batch_time))
    all_probs = torch.cat(all_probs).cpu()  # concatenate probs from all batches, move to CPU
    all_uuids = [uuid_index_mapping[i] for i in all_uuids]  # convert UUID indices to UUIDs

    # Create dataframe and store as CSV
    df1 = pd.DataFrame({'UUIDs': pd.Series(all_uuids)})
    df2 = pd.DataFrame(all_probs.numpy())
    df = pd.concat([df1, df2], axis=1)
    csv_path = './TESTRESULTS__' + args.checkpoint.split('/')[-1] + '__' + args.h5data.split('/')[-1] + '.csv'
    df.to_csv(csv_path, header=False, index=False)
    print("\nSaved results to {0}\n".format(csv_path))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
