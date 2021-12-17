#!/usr/bin/python3

import time
import torch
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from utils import *
from torch.utils.data import DataLoader, TensorDataset


parser = argparse.ArgumentParser()
parser.add_argument('--nn-type',  '-t', help="network to use", default="fc")
parser.add_argument('--epoch', '-e', type=int, help="network to use", default=10)
parser.add_argument('--lr',   '-l', type=float, help="network to use", default=0.001)
parser.add_argument('--ld',   '-d', type=float, help="network to use", default=0.90)
args = parser.parse_args()


def main():
    # General setup
    print('General setup ...')
    n_epochs = args.epoch
    lr = args.lr
    nn_type = args.nn_type
    ld = args.ld
    seeds_rngs()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set the random number generator
    print('Load and prepare data ...')
    all_X, all_y = load_from_disk('all_images', 'all_labels')
    if args.nn_type == "cv":
        all_X = all_X.reshape(-1, 1, 28, 28)

    # Splitting data
    train_X_y, valid_X_y, test_X_y = split2tvt(all_X, all_y)
    del all_X, all_y

    # Construting datasets
    print('Construting datasets ...')
    trainset = DataLoader(train_X_y, batch_size=64, shuffle=False)
    validset = DataLoader(valid_X_y, batch_size=64, shuffle=False)
    testset = DataLoader(test_X_y, batch_size=64, shuffle=False)

    # Defining neural networks
    print('Defining neural networks ...')

    if args.nn_type == "fc":
        fc_layout = (784, 64, 64, 10)
        net = FCNet(fc_layout, init="he").to(device)
        net.load_state_dict(torch.load('models/net_fc'))
    elif args.nn_type == "cv":
        cv_kernels = ((4, 4), (2, 2))
        cv_strides = ((1, 1), (1, 1))
        cv_padding = ((2, 2), (1, 1))
        cv_layout = (1, 32, 64) # 28, 14, 7
        cv2linear_size = (7, 7)
        out_classes = 10
        net = ConvNet(cv_layout=cv_layout, cv_kernels=cv_kernels, cv_strides=cv_strides, \
                cv_padding=cv_padding, cv2linear_size=cv2linear_size, \
                out_classes=out_classes).to(device)
        net.load_state_dict(torch.load('models/net_cv'))

    # Evaluate neural network
    print('Starting evaluation ...')
    start = time.time()
    print(f'\nTraining accuracy: {fc(net, trainset)}, Validation accuracy: {fc(net, validset)}, Test accuracy: {fc(net, testset)}\n')
    end = time.time()
    print('Finishing evaluation ...')

    print(f'Evaluation time: {end-start}')

if __name__ == "__main__":
    main()
