from __future__ import print_function
import argparse
import shutil
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from wresnet_models import *
from h5_dataloaders import *

parser = argparse.ArgumentParser(description='SETI Classifier - Train Model')

parser.add_argument('arch', metavar='PATH',
                    help='architecture to use')
parser.add_argument('h5data', metavar='PATH',
                    help='path to hdf5 file with training and validation data')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4 * 3, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

best_acc = 0
epochs_since_improvement = 0

# Available models
# model_archs = ['resnet18', 'resnet34', 'resnet50', 'resnet86', 'resnet101', 'resnet131', 'resnet203', 'resnet152',
#                'resrnn2x2', 'resrnn2x3', 'resrnn3x2', 'resrnn3x3', 'resrnn3x10', 'wresnet28x10', 'wresnet16x8',
#                'wresnet34x2', 'wresnet40x10', 'wresnet28x20', 'densenet161', 'densenet201', 'dpn92', 'dpn98',
#                'dpn131']
model_archs = ['wresnet34x2']

classes = ['brightpixel', 'narrowband', 'narrowbanddrd', 'noise', 'squarepulsedn', 'squiggle',
           'squigglesquar']
target_class_index_mapping = {}
for i, c in enumerate(classes):
    target_class_index_mapping[c] = i


def main():
    """
    Load model's graph, loss function, optimizer, dataloaders.

    Perform training one epoch at a time, validating after each epoch. Each epoch's model is written to file,
    and the best model thus far is written to a seperate file.
    """

    global args, best_acc, epochs_since_improvement
    args = parser.parse_args()

    print("\n\nChosen args:")
    print(args)

    assert args.arch in model_archs
    model = eval(args.arch + '()')
    print("\n\nMODEL ARCHITECTURE:\n\n")
    print(model)

    model = torch.nn.DataParallel(model).cuda()  # data parallelism over GPUs

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Normalizer
    h = h5py.File(args.h5data, 'r')
    mean = torch.FloatTensor(h['mean'][:])
    mean = mean.permute(2, 0, 1)  # permute to feature dimensions first
    std_dev = torch.FloatTensor(h['std_dev'][:])
    std_dev = std_dev.permute(2, 0, 1)
    h.close()
    normalize = transforms.Normalize(mean=mean,
                                     std=std_dev)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        h5Dataset(args.h5data, [2, 3, 4, 5], target_class_index_mapping, transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        h5Dataset(args.h5data, [1], target_class_index_mapping, transforms.Compose([normalize])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):

        # Halve learning rate if there is no improvement for 3 consecutive epochs, and terminate training after 8
        if epochs_since_improvement == 8:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
            adjust_learning_rate(optimizer, 0.5)

        train(train_loader, model, criterion, optimizer, epoch)

        acc = validate(val_loader, model, criterion)

        is_best = acc > best_acc
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Perform one epoch's training.
    """

    batch_time = AverageMeter()  # forward prop. + gradient descent time this batch
    data_time = AverageMeter()  # data loading time this batch
    losses = AverageMeter()  # loss this batch
    top1 = AverageMeter()  # (top1) accuracy this batch

    model.train()  # train mode

    start = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - start)

        input_var = torch.autograd.Variable(input).cuda()
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        acc = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(acc, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Perform validation after each training cycle.

    Returns:
        top1.avg (float): Average accuracy on the validation data
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()  # eval mode

    start = time.time()

    for i, (input, target) in enumerate(val_loader):

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        output = model(input_var)
        loss = criterion(output, target_var)

        acc = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(acc, input.size(0))

        batch_time.update(time.time() - start)
        start = time.time()

        if i % args.print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print('\n * Accuracy {top1.avg:.3f}\n'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best):
    """
    Saves model state to a checkpoint.

    If this is an improved model, also save to a seperate file.
    """

    filename = args.arch + '_batchsize' + str(args.batch_size) + '_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'BEST_' + filename)


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


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %.3f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(output, target):
    """
    Computes accuracy, from predicted and true labels.
    """

    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    correct_total = correct.float().sum()
    return correct_total * (100.0 / batch_size)


if __name__ == '__main__':
    main()
