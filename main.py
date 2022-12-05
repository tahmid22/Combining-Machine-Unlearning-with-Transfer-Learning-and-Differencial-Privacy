import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torchvision.transforms import ToTensor
#import resnet as models
from sisa import SISA
import numpy as np

torch.cuda.empty_cache()

'''
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# parser.add_argument('--world-size', default=1, type=int,
#                     help='number of distributed processes')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='gloo', type=str,
#                     help='distributed backend')
parser.add_argument('--group-norm', default=0, type=int,
                    help='number of channels per group. If it is 0, it means '
                    'batch norm instead of group-norm')
parser.add_argument('--shards', default=1, type=int,
                    help='number of chards')
'''

def main():
    global args
    #args = parser.parse_args()

    cudnn.benchmark = True

    train_transform = tt.Compose([
        tt.RandomResizedCrop(224),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
    ])

    val_transform = tt.Compose([
        tt.Resize(256),
        tt.CenterCrop(224),
        tt.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=tt.ToTensor(),
    )

    val_test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=tt.ToTensor(),
    )

    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [5000, 5000])

    sisa_1 = SISA(train_dataset, val_dataset, test_dataset, shards=10)
    sisa_1.fit(epsilon=3, fine_tune_percent=1, fine_tune_method=1, batch_size=128, epochs=200, workers=16)


    # X_train = train_dataset.data
    # y_train = np.array(train_dataset.targets)
    # X_test = val_dataset.data
    # y_test = np.array(val_dataset.targets)
    
    # # train_loader = torch.utils.data.DataLoader(
    # #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    # #     num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    # # val_loader = torch.utils.data.DataLoader(
    # #     val_dataset, batch_size=args.batch_size, shuffle=False,
    # #     num_workers=args.workers, pin_memory=False)

    # sisa_1 = SISA(X_train, y_train)
    # sisa_1.fit(args)

    # for i, (input, target) in enumerate(train_loader):
    #     print(type(input))
    #     print(type(target))
    #     if (i==10):
    #         break

if __name__ == '__main__':
    main()
