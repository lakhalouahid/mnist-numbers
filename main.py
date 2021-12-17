#!/usr/bin/python3

import time
import torch
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from utils import *
from torch.utils.data import DataLoader, TensorDataset


parser = argparse.ArgumentParser()
parser.add_argument('--nn-type',  '-t', help="network to use", default="cv")
parser.add_argument('--epoch', '-e', type=int, help="number of epochs", default=10)
parser.add_argument('--lr',   '-l', type=float, help="learning rate", default=0.001)
parser.add_argument('--ld',   '-d', type=float, help="learning decay", default=0.96)
parser.add_argument('--visualisation',   '-v', action="store_true", help="Enabl Data visualisation")
args = parser.parse_args()


def main():
    # General setup
    print('General setup ...')
    n_epochs = args.epoch if args else 10
    lr = args.lr if args else 0.001
    nn_type = args.nn_type if args else "cv"
    ld = args.ld if args else 0.96
    enable_data_visualisation = args.visualisation if args else True
    forceReloadData = False
    seeds_rngs()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set the random number generator
    print('Load and prepare data ...')
    filenames = ['all_images', 'all_labels']
    if check_files(*filenames) and not forceReloadData:
        all_X, all_y = load_from_disk(*filenames)
    else:
        all_X, all_y = load_mnist_data()

    # Reshape for conv nn
    if args.nn_type == "cv":
        all_X = all_X.reshape(-1, 1, 28, 28)

    # Dataset visualisation
    if enable_data_visualisation:
        display_samples(all_X[:20*30], shape=(20, 30))


    # Splitting data
    train_X_y, valid_X_y, test_X_y = split2tvt(all_X, all_y)
    del all_X, all_y

    # Construting datasets
    print('Construting datasets ...')
    trainset = DataLoader(train_X_y, batch_size=64, shuffle=False)
    validset = DataLoader(valid_X_y, batch_size=64, shuffle=False)
    testset = DataLoader(test_X_y, batch_size=64, shuffle=False)

    # Defining model
    print('Defining neural networks ...')

    if args.nn_type == "fc":
        fc_layout = (784, 64, 64, 10)
        net = FCNet(fc_layout, init="he").to(device)
    elif args.nn_type == "cv":
        cv_kernels = ((4, 4), (4, 4), (2, 2), (2, 2), (2, 2))
        cv_strides = ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))
        cv_padding = ((2, 2), (1, 1), (1, 1), (1, 1), (1, 1))
        cv_layout = (1, 8, 16, 32, 64, 128) # 28, 12, 6, 3
        cv_maxpool = [True, False, False, True, True]
        cv2linear_size = (1, 1)
        out_classes = 10
        net = ConvNet(cv_maxpool=cv_maxpool, cv_layout=cv_layout, cv_kernels=cv_kernels, cv_strides=cv_strides, \
                cv_padding=cv_padding, cv2linear_size=cv2linear_size, \
                out_classes=out_classes).to(device)

    # Optimizer and cost function
    print('Defining optimizer and cost function ...')
    loss_func = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    # Training all
    print('Starting training ...')
    start = time.time()
    loss_history = train_net(net, trainset, validset, loss_func=loss_func, optimizer=optimizer, n_epochs=n_epochs, batch_log=64, learn_decay=ld, debug=False, debug_epoch=8)
    end = time.time()
    print('Finishing training ...')

    # Loss history
    f, a = plt.subplots()
    a.plot(loss_history.detach().numpy())
    f.canvas.draw()
    f.canvas.flush_events()
    plt.show()

    print(f'\nTraining accuracy: {fc(net, trainset)}, Validation accuracy: {fc(net, validset)}, Test accuracy: {fc(net, testset)}\n')
    print(f'Training time: {end-start}')

    # Saving model
    if input("Do you want to save the model (y/n): ") == "y":
        if args.nn_type == 'cv':
            torch.save(net.state_dict(), 'models/net_cv')
        elif args.nn_type == 'fc':
            torch.save(net.state_dict(), 'models/net_fc')

if __name__ == "__main__":
    main()
