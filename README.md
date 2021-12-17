# Python wrappers around Pytorch for fast prototyping

## About

this repo is used for teaching me, about pytorch, i decided to wrappe pytorch
functions, and all the load and the preparations of datasets, in wrappers to hide
some of the complexity of those functions. refers to the code, to know more.  

## Contents

This repository contains:

- `data` folder containing all the dataset of mnist for classification of numbers
- `models` folder containing the trained models
- `html` folder containing the saved sessions of some of the jupyter qtconsole runs in html format
- `utils.py` python module for all the functions
- `main.py` python script for running the training, and supporting options
- `notebook.py` python file that attended to run in jupyter qtconsole

### Feel free to clone this repository

## This is notebook example of train the mnist dataset numbers
<div>

<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">In [</span><span style=" font-weight:600; color:#000080;">2</span><span style=" color:#000080;">]:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">[remote] In [</span><span style=" font-weight:600; color:#000080;">1</span><span style=" color:#000080;">]:</span> #!/usr/bin/python3</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Imports</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> import time</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> import torch</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> import torch.optim as optim</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> import matplotlib.pyplot as plt</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> from utils import *</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> from torch.utils.data import DataLoader, TensorDataset</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Set general configurations about the number of epochs, learning rate,</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% neural network type (Convolutional / full-connected), and visualisation</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Also, we need to seed the random number generators</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print('General setup ...')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> n_epochs = 88</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> lr = 0.001</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> nn_type = &quot;cv&quot;</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> ld =  0.97</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> visualize_data = True</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> forceReloadData = False # force Reloading of all the data from raw files</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> seeds_rngs()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Configure the training of the GPU:0, if not available do it on the CPU</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> device = torch.device(&quot;cuda:0&quot; if torch.cuda.is_available() else &quot;cpu&quot;)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print(device)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Start loading and preparing the train set, the validation set and the</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% test set, the data is all ready prepared in the all_images.npy and </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% all_labels.npy in the folder `data`.</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% if there is any need for loading the data from the raw file, we use</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% the wrapper function load_mnist_data to do that</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print('Load and prepare data ...')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> filenames = ['all_images', 'all_labels'] # no need to add '.npy' extension</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> if check_files(*filenames) and not forceReloadData: # we check if files exist</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     all_X, all_y = load_from_disk(*filenames)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> else:</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     all_X, all_y = load_mnist_data()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Reshape the data if we using a Convnet from (-1, 28*28) to (-1, 1, 28, 28)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> if nn_type == &quot;cv&quot;:</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     all_X = all_X.reshape(-1, 1, 28, 28)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Visualise some of the dataset</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> if visualize_data:</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     display_samples(all_X[:20*30], shape=(20, 30))</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Splitting the data to trainset, validset, testset</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> train_X_y, valid_X_y, test_X_y = split2tvt(all_X, all_y)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # We don't need any more those arrays</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> del all_X, all_y</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Construting datasets wrappers</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print('Construting datasets ...')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> trainset = DataLoader(train_X_y, batch_size=64, shuffle=False)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> validset = DataLoader(valid_X_y, batch_size=64, shuffle=False)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> testset = DataLoader(test_X_y, batch_size=64, shuffle=False)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # We don't need any more those arrays</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> del train_X_y, valid_X_y, test_X_y</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Defining  the model of the neural network</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print('Defining neural networks ...')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> if nn_type == &quot;fc&quot;:</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     # if we want to use a fully connected neural network</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     fc_layout = (784, 64, 64, 10) # (input, fist hidden, second hidden, output)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     # we pass the layout to the FCNet function that do the magic</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     # we use he initialisation as a initialisation strategy</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     # the training is being done in the device</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     net = FCNet(fc_layout, init=&quot;he&quot;).to(device)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> elif nn_type == &quot;cv&quot;:</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     # if we want to use a convolutionnal neural network</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     cv_kernels = ((4, 4), (4, 4), (2, 2), (2, 2), (2, 2))</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     cv_strides = ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     cv_padding = ((2, 2), (1, 1), (1, 1), (1, 1), (1, 1))</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     cv_layout = (1, 8, 16, 32, 64, 128) # 28, 12, 6, 3</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     cv_maxpool = [True, False, False, True, True]</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     cv2linear_size = (1, 1)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     out_classes = 10</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     net = ConvNet(cv_maxpool=cv_maxpool, cv_layout=cv_layout, \</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>             cv_kernels=cv_kernels, cv_strides=cv_strides, \</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>             cv_padding=cv_padding, cv2linear_size=cv2linear_size, \</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>             out_classes=out_classes).to(device)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Setting of the optimizer and cost/loss function</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print('Defining optimizer and cost function ...')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # multi-label one-versus-all loss based on max-entropy</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> loss_func = nn.MultiLabelSoftMarginLoss()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # adam optimizer</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> optimizer = optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.999), \</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>         eps=1e-8, weight_decay=0, amsgrad=False)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Start the training of the model</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print('Starting training ...')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> start = time.time()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> loss_history = train_net(net, trainset, validset, loss_func=loss_func,</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>         optimizer=optimizer, n_epochs=n_epochs, batch_log=128,\</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>                 learn_decay=ld, debug=True, debug_epoch=8)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> end = time.time()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print('Finishing training ...')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print(f'Training time: {end-start}')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Plot the loss history of the model</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> loss_history = loss_history.detach().numpy()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> f, a = plt.subplots()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> a.scatter(np.arange(max(loss_history.shape)), loss_history)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> a.set_title('the cost value of the model during the training')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> a.set_xlabel('the training time in term of epoch numbers of training')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> a.set_ylabel('the value of the softmax error')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> f.canvas.draw()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> f.canvas.flush_events()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> plt.show()</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Evaluating the model on the dataset, validset, testset</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> train_loss = fc(net, trainset)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> valid_loss = fc(net, validset)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> test_loss = fc(net, testset)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print(f'\n\nComputing the performance of the modele...')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> print(f'\nTraining accuracy: {train_loss},\</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>         Validation accuracy: {valid_loss},\</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>         Test accuracy: {test_loss}\n')</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> </p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> # %% Saving model</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> total_loss_str = str(train_loss) + '_' + str(valid_loss) + '_' + str(test_loss)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> if nn_type == 'cv':</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     filename = 'models/net_cv_' + total_loss_str + str(time.time())</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> elif nn_type == 'fc':</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span>     filename = 'models/net_fc_' + total_loss_str + str(time.time())</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">            ...:</span> torch.save(net.state_dict(), filename)</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" color:#000080;">In [</span><span style=" font-weight:600; color:#000080;">3</span><span style=" color:#000080;">]:</span> setup ...</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">cuda:0</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Load and prepare data ...</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><img src="data:image/png;base64,
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
sLCQnj17RvX19XTlyhWysbGRuOaHDh1KvXv3Jnd3d8rOzqaYmBiJ+yaXyyVD
Q0NycHCgWbNm0bx58yg4OJg2b95MioqKUvvC3d2d/P396eLFi/Ts2TM6ceIE
mZqais1jbGxMqampdP/+fTp27Bht376djh49SoGBgWLzqKio0IULF8jW1lZs
GrHnyu89dLy8vOju3bukpqYmsRM4HA75+/tTRkYGFRUVUUNDA+Xl5VFkZCSt
WrWKpk6d2mUxyMnJkZOTE6moqJCbmxvFxcXRpEmTpE5cFxcXunDhArm4uNC8
efMkHlLq6uoUGxtLNTU1tH79euHC69gEs7OzydXVVWz+Xr160d69e+nAgQO0
Z88eWr16tdT6CQQCGjVqFFlYWFB4eDgtWLBA4oJSVlamhIQE8vX1JX19fbKw
sKDTp0+Tvb29xHJGjx5Nb9++pZaWFkpOTqalS5eSs7Mz/fLLL5Senk7q6upi
y+vY8OXk5Ojq1au0bNkyiWXNmTOH3r59SxMnTiQAZGRkRKWlpbRz506p/cFi
sWjUqFFUVlZGra2ttGPHDrGLccWKFZSVlUU1NTXU2NhIK1asIBMTEwoODqZX
r15RWVkZLV68uNPmGRoaSpcuXSIdHZ0u3zM1NaXk5GQaOHBgl98mT55MBQUF
NHjwYOLz+aSurk4bNmyg1NRUkRt3B1lZWVF2djY1NDRQVlYWrV69mmJjY2nD
hg1SN04Oh0McDoc8PT0pKytL7OZuZmZGR48epbi4OEpKSqKXL1/Smzdv6P79
+zRlypROh27HPI2OjqZHjx7RwYMHadKkSeTt7U3+/v60fft2io6OptTUVHJ2
dpa4Vk6fPk0eHh5Sx/TDNezr60u1tbW0adMmmfN19MXIkSMpJyeHpk6dKjW9
sbExZWZm0p07d2j58uU0YsQIsrKyIgUFBZEXNC6XSzwejwQCAfXr14/2799P
OTk5ZGdnJ/L7fD6fzpw5Q0lJSeTi4kJGRkZkampK06dPp9LSUoqJiSE+ny/T
fF+3bh1dvHixyzgBIAUFBfLx8aGYmBgqKCigpqYmamtro/b2dmpvb6empiZy
d3eXWIaGhgYlJyfT9evX6eeffyZHR0eRl7gOYhiG1q5dS0+fPiUrKyvhZaG+
vp6mTZsmca7fu3dP7F4C/AsOnWnTplF6errEQ4fH49HKlSupqqqKkpOTKTIy
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
lStXyNHRsVvtjI6OlpiGYRgyNTWlmJgYioyMlEmY/zE5OzvThQsXyNLSUqzg
t0P9sba2lpqamqikpIQWL14sVbtm3LhxVFJSQmlpaZScnExFRUWUl5dHQ4cO
ldh/GzZsoLKyMjp//jxdvnyZ0tLSyMfHR+TtlGEY8vX1pbdv39KLFy9EvkIk
kYmJCeXk5ND169fp3LlzElXHWSwWeXl5UVpaGt25c4eSk5MpKSmJ8vPz6ciR
I6SjoyN2fOXl5Sk0NJTy8vIoJSWFcnJyyMvLS6TGUQetWrWKrl27Rn5+frR5
82ZKT0+n+Ph4iS/LwYMHU05ODu3cuZO0tbWJYRhSVlYmVVVVsbf7DrKysqL8
/HwaP358t/pw2rRpVF1dTQsXLpQpPY/HI1dXV1q2bBktX76chg8fLvMrYtq0
aVRZWSlWK+xD8vPzo9raWnr9+jVVVlZSamoqDR06VOIanDVrllAF+dKlS/T4
8WP66aefxK4N4H814/z8/Mjf35/OnTtHISEh5OrqKpGDMHz4cCorK6OQkBA6
duwYFRUVUVZWllgNV0VFRfLy8qIxY8aQvr6+xLkjjgwNDengwYP05s0bIYdp
5MiRItNdvHiRfvnlF7p69SqFh4dLVaTq6Iu7d+9SVlYWpaSkUG5ursyKGIqK
ihQYGEjp6el09epVkfuzuHOFkXTTEecGR1aYmZnB1dUVhYWFaGxsRE5ODmpq
aiTyGdlsNvr06YNPP/0UmpqayM7Oxv3798Xy5j+GtbU1fHx8YGJigoULF0o1
6Ou47QwcOBB9+vRBVVUV4uPjZTYEBN7LKIyNjfHbb7+JTaOpqYmNGzeiqKgI
Bw4c+F1W7tbW1jh58iRu376NlStXipQf6erqwt/fH8OHD4e2tjaOHTuGjRs3
Sn1RWVtbQ19fH7du3UJ7ezt0dHTwt7/9DUlJSRL7QiAQwM3NDUZGRigvL8ed
O3fw/PlzsTdoeXl5eHp6ory8HGlpaWJvlqIgJyeH48ePw8PDAz/++CO+/fZb
iXOJYRhoamrC0dERzs7OyMjIwMuXL5GWliZWxvJhPZ2cnKCmpoY7d+6gsLBQ
4qvAwMAAq1atgrGxMVpbW3H16lWcP39eonJExw29qampWx4c5OXlceTIESgp
KWHMmDEyGTED78fq9OnT4PP5mDp1qlSZxx/FuHHjMGvWLEyePFlqfysqKsLd
3R16enqoq6tDXFycVPkFm81Gv379YGlpCeC9IlFWVpbEOcXn87FhwwYYGhoi
OTkZSUlJyMvLk9qHHA4Hc+bMwaxZs1BXV4dr167hwIEDEuf6nwEFBQX87W9/
w+zZs2FnZ4cxY8aIfFGYmJhg+PDhKCsrw82bN2XyFMMwjNAgV0dHB2vXrsWF
Cxdkbg+Hw4GcnBza2tpEvpDEucH53SrT/6kkLy9PHh4eIm0z/p3EMAzx+fxu
8d4/Jg6HQw4ODhLbxjAMcTgc4vP5JBAIpN6Y/9tIW1ubvvrqq27x7AH8rptm
d4nNZhOPxyMul/uHxlka6enpUUxMDI0ZM6bb9bOwsKBevXr9JWPFYrFksl/5
q6nDJur3tEcgEBCfz5dJdvtn11kgEPzp84phGKHN0p/97X/LS6cHPejBvwYc
Dgft7e3/1X7JevB/G+JeOj2hDX4nWCwWFBUVoaCgIFQo+L8ADocDU1PT/1Nt
At6zHR0cHGBmZvbvrsqfgra2tp4Dpwf/nfg97DUjIyOysrKS6TnWIVDbsWMH
RUdH0759+8jBweF3PzHl5eVF+ivT19cnS0tL0tPTk+qb7GNis9lkYGBAVlZW
ZGxsLPXpzWKxaN68eRQXF0enT5+mnTt3/i7FgN9LAoGA3N3dadu2bRLVIlks
FpmZmdGwYcNo6NChMrE6LC0tKSYmptvsqw/7sl+/fjKVpaurS7a2tv9SVgXD
MDRq1CjKzMyk1tZWKiwsJBcXF5nyamho0IYNG+jatWt06NAhMjQ0lLlcLS0t
mj9/Prm6uopVnPkj429lZSVUk2Wz2TL5TxM3RywsLGjw4MFS81tbW1NYWJhM
qrjAexV3d3d32rx5M507d07qumexWKSrq0sDBw6kfv36/ctZWFwul+bNm0cm
Jibdzuvo6Eh+fn4S+0xHR4c2btxI8fHxFBsbS2vWrCFnZ2eZ2yUQCGjIkCHk
6OgokyrzH+mH4cOHU2RkJG3ZsoW0tLT+lO/+aW5wXF1dKS0tjW7fvi3TxuTm
5ka5ubmUmJhIly5douzsbLpx44ZUR5fA//Kgvb29aceOHXTp0iVKTk6mqKio
TmV/qEWVl5dHiYmJFBQURP7+/jRkyBCxg8xms2n27Nl0+/ZtevLkCZWXl1Nx
cTGFhYWRsbGxxMUxefJkmj59Orm5udH9+/fpxo0b1Lt3b4ntYRiGWCwW6enp
kbm5Odna2pKbm5tUewzgvV8mY2NjGjlyJF24cIFevHhBW7duFatxw2KxyNvb
m/Ly8qikpIQyMjLIy8uLZs6cKbHvp02bRjt37pR5YSgrK5O/vz8ZGRkRANLU
1KSUlBShnZAk2rVrF+Xm5srsVqWDlJSUhHYVX331lcS0PB6Pzp8/T/Pnzyd/
f3/Ky8ujhIQEqTIeAwMDOn36NJ06dYr8/f0pPj6etm7dKlO/aGlpUWRkJJWU
lFBSUhJFR0eLnE+mpqY0dOhQMjAwIBUVFTI3Nydra2upG4yDgwM9e/aMfvjh
XJDLdQAAIABJREFUB2IYhgYOHEjXrl2TOv8+nIcGBgY0YcIEioqKokePHtHW
rVul9sn48eMpNzdXpk3JxMSEwsLCaPv27TRv3jzy9vamgwcPCufJxyQvL09L
ly6lmzdv0t27d+n27ds0ZswYmS6QbDabHB0dKTw8nCIiIiSu3Y/HODk5Web0
HWRra0t3796ladOmSTx0XF1dqaSkhDZt2kTbtm2ja9euUVFREZmZmUktg8Vi
0cSJE6m6upqys7MpISFBpnyixprL5dKIESPIwsKiy++KiooUFBREV65cof37
91N0dDRt2rRJYrt4PB45OzvTqVOn6ObNm3T9+nUaMWJElzx/2qHz9ddfU2lp
KWVmZsqklqekpERGRkYkLy9PysrKFBQURImJiVJ9KrFYLPL396fCwkJ68OAB
hYeH0/r16+mrr76iAQMGdLlJ6+rq0uDBg2nw4MHk4OBA06ZNo/v371NoaKjY
l4uTkxOVlpbSuXPnyM/Pj2bMmEE7d+6k8vJySklJkegLrEMAx+FwaPTo0VRS
UkL9+/cXmVZRUZFGjx5NW7ZsoYSEBHry5AmVlpZSZWUlNTU10aJFi6Qa6c2f
P58eP35Mz549o8jISKkvFzMzMyooKKCDBw+SmZkZeXl5UVFREd28eVPiZSEq
Koq8vb1lntQTJkygqqoq4SHT4W9Lks+zDjp8+DC1traSr6+vTGWx2WwaPnw4
RURE0NatW2n58uX0008/SXVoqKOjQywWS9iPxcXFEm+3HA6Htm/fTr/88ovw
gP7qq6/o4sWLUl9w2traFBkZSWVlZTR9+nTy8fGhV69eiVStXb9+PVVUVFBO
Tg6lp6dTUlISpaenU3BwsMTDzdvbm5qamigyMpJYLBbNnDmTUlNTZboN8/l8
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
wcE0evRo0tHRkXqoMgxDw4YNo5SUFNq+fbvES6Cvry+dOHGCzMzMyMbGhtLT
0yk+Pp7c3Nykts3Pz48aGhpowYIFUtNaW1tTYmKicKMtLi6mt2/fUl5entTI
wSoqKkKv1vv27aOCggLau3dvF0UYY2NjOnr0KEVGRtKCBQvI3t6ePD09yd/f
nwICAmjRokXk4uLSZQ3379+fTpw4Qf379yd1dXWSk5MjfX192rRpE82YMUPq
+H5oDC+rxuyePXto/fr15OrqSqdOneqWUoW6ujotW7aM0tPTqaSkhF68eEFl
ZWUUHR0tUeN47dq1Yh32/iGHn6NHj8b48eNRVFSEgICATq4wysvLoaurK2QX
fQgFBQXEx8cjKioK9fX1uH//Puzs7ODk5ARVVVWxcbzl5eWxcuVKpKSkIDIy
Er169cLw4cNlqSoYhkFDQ4NUdx0fQktLC6tWrcLp06dx+fJliWlbW1tx6NAh
cLlc9O3bF7Nnzxb6y5IF1dXVqKioQGFhITw8PGTyUWZqagoHBwdYWlrC1NQU
+vr6ePToEU6fPo3U1NROaRmGgY+PDwoLC7Fv3z40NzdDR0cHACSyoWbOnInP
P/8cU6ZMQUZGBjgcDn755RcEBATA19dXohFca2srHjx4AGdnZ3A4HKnGihcv
XhSytzgcDjw8PFBcXCzWaefHYBgGEydOxKtXr7B//36RjlY78OmnnyIgIAA3
b97EokWL8M0332DUqFEwMTGBq6srjh49KrGsAQMGSPTf9zFGjBiBuro67Nq1
C19++SWUlZWlxqvX1NQUOsaVBU5OTli7di1ycnLA4/Hw+eefo6KiQug3UBwc
HBzQ2NiI77//HkePHkVzc7NUFkwHC8rNzQ1eXl44d+4cQkJChL+JYlOyWCzU
1NSgb9+++P7771FUVITNmzdLXSOKiooYNmwYMjMzcerUKQDvWZUd/h4/Rmlp
KcLCwtCnTx+0tbWhtrYW8+bNQ11dnURjYwUFBWzfvh12dnbYsmULoqOjMWfO
HGzcuBFRUVGIi4sTpp0wYQJ8fHzQ2tqKL774AvX19WhsbERzczMaGhrA5/PB
5XLh4ODQqczs7GzMnTu3kzF3aWkpDh8+jK1bt+LkyZNi2ahKSkowMzPDkydP
IC8vj1mzZqG5uRmHDh2SuLYqKythbW2No0ePIigoSCabPgCwsbHBDz/8gEGD
BuHs2bNoa2tDZWUlnjx5gqSkJLEOglksFvr374+0tDSZyhFClpeOqakpOTg4
iNQvHzt2LG3ZskXk09nKykok22nt2rUSY3IMHDiQtm7dKvymnp4erVy5UqZb
tIKCAp0+fVrmoGqKioq0d+9eOnPmDKmoqJCCgoLEvBoaGjR//nyKiIig8vJy
am1tpZKSEjp06JBMrwKGYSggIIDevn1LJSUlIl3ef0gdbk/i4uJo9erV5OLi
QgYGBmJZFQzDkJ+fH+Xk5NDPP/9Me/fupdu3b1NGRoZE+6O1a9fS06dPaffu
3TR58mTq168frVu3jl6+fEm2trZS27RixYrfZWfi4+NDz54965Z9j6mpKWVm
ZtKkSZMk3gDZbDYdOHCACgoKhGxMDQ0N6t27N8XGxkq9cXK5XIqMjKRJkybJ
XLegoCAKCAggS0tLKizWDa7IAAAgAElEQVQsJA8PD6l5pk2b1i37Ej6fT1pa
WsTlcklJSUloXyGtLzpYSSkpKeTr6yvVZq6jD6KioqixsZGePn1KkZGRtHbt
WoqJiREb30VNTY2CgoLo7du3dO/ePTIzM5Pppj5p0iR69eqVkNtgbGxMe/fu
lRoKoOMlYGtrS2/fvqVdu3ZJjKG1fPlyys/PJ3d3d2G9Zs6cSbW1tV3G2t7e
ngIDA2ns2LHk6upKAwcOFIZqUVBQoO+++45SUlJkig8GvFfrvnDhgkTWl5aW
Ft26dYsMDQ3J19eXCgoKKDs7Wyq7VkdHh06cOEGZmZlUVFQkkgvyMfXr148y
MjKovb2dQkJCiM/nE5vNljlAZ0xMDA0cOFDk73/opVNQUCDS4pTD4cDOzg5n
zpwRKWh9+vQpLCwsYGVlhcLCQjQ3N0NXVxeZmZkSb386OjpQUVEBm82GgoIC
1q1bB4FAIJNrlfb2dqipqUFZWVnqDZVhGMycOROurq6YP38+dHV18c9//hP6
+vqYOnWqSKG4vr4+7Ozs8O7dO9y4cQPAe2+tAwYMwMaNG/H3v/9dovBTV1cX
rq6uICKsWbNGqlW4lZUVBAIBVq9ejYyMDKkCbSLCyZMnYWFhgYULF6K8vFyo
huvt7Y1du3aJvDUGBQUhIyMDX375JWbMmAEdHR0YGxuDy+XCy8sLWVlZEm/F
3RXOs9lsfPHFF9i4cSMOHz4s9YXZARaLha+//hqvXr1CXFycxDL79u0Ld3d3
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
cwwfPhzp6ekYOXIk9PT0IBAIUFtbi6CgILnMEVKGkpqaGjQ2NqKyshJcLhd3
797FuXPnlPaHz+ejS5cumDJlCjp37gxTU1PweDw0NDQgNDRU+TtjetNxcnKi
kydPkrGxMY0bN44ePXpEu3btoq5du6o8BXbr1o0qKiooPDxc5clCKBRSdHQ0
Xb9+ne7cuUOpqamUnZ1NjY2NKtMHdOjQgVJTU+ny5cvk7u5OgwcPpufPn9OE
CRNatWttbU2PHj2ihoYGev36Nf32229069YtKioqovLyciosLKTCwsI23k4c
Doe8vLzanCYlEgklJCSodKHcuHEjLV68mACQqakpdenSReXNZcCAAXT16lUy
NjYmS0tLxt5bXC6XevbsSbdu3aKCggLKzs6mjIwMOn36NM2cOVOpK+qQIUNo
165djNrx8fGhLVu2UKdOncjNzY0uXLhAAQEBKhkMxGIxubi40PHjx8nX11et
09+4ceNUuoN+LFJuLg6HQ66urnT69GnavXt3m1O0UCikDRs2UFBQEBkbGxOH
wyEej0f79++nWbNmqZy3Z8+epdDQUPLy8qLr16+rZJwQi8V0+vRpKiwspAsX
LlBaWhq9evWKMjMzVd4ybW1t6dSpU/TTTz+RlpYWcTgcxiECM2fOlM1DVSKR
SGjq1Kl05MgRmjFjBg0cOJAuXryo8Kapq6tLFy9epD179sjG18/Pj969e0cT
J05U2A6Px6Pk5GRKTEykLVu2UFJSEp0/f16lR6mBgQEdOHCAamtrKS8vjxoa
Gqi5uZkKCwvlpspgs9m0du1alWzyH8+dmJgYevv2bat5yuVy6ejRo1RZWaky
7Yquri6tWrWKfvvtN8rOzqbHjx9TUVER/fDDDwrraGlp0caNG+nKlSuyEBBl
nHJisZgMDQ1JIpGQiYkJGRkZkY+PD+Xn58vlNWSz2XT48GG6fPkyDR48mHr2
7EkWFhZkbGyscsy1tLRo5cqVVFJSQlu3bqU+ffpQSEgI3bhxg+7evSvznPvT
N52ysjKUlZXh2LFjsLOzw6JFi7B7926VzLhaWlqYMWMGHjx4gIMHD4LNZstO
Wvfv329Tvr6+Ht9//z2EQiF4PB4aGxuxd+9eiMViPHr0SGE7ZmZmWLZsGRob
GzFq1CiUlpbi1atXaGlpgb+/P3bt2iVjWebxeKisrMSjR4+Ql5eH2tpavHz5
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
gsCIiAieOHGCenp6Tf6+ZMkSbt68udEzAwMDbty4kffu3ePZs2f58uVLnj59
msnJySwoKGh0uvynlA7w9tjq7u5OV1dXheYqVTxq1ChmZ2ezX79+9PHx4ddf
f80ffviBoaGhCj+2gb/44gsuW7asWW0NHTqU+fn59PPz08hEERUVxczMTL5+
/ZqvX79meno6IyMjaWtrq7S+tbU1hw4dyi+++IL+/v7NEogG7t27N2fMmKGy
TKdOnZiZmckDBw5w/fr1Gvnh29ractWqVTx69CgjIyPlJ9Tmvp8mLJFIeODA
Aebk5HD27Nm0tbWliYmJRuaDTp068datW82KWDc0NKSXlxfd3NyatWP8Z9jO
zo737t1jRESEynLa2tqMjY3lqVOneOHCBV69erXJJuuPLAgCx40bxyNHjjA2
Nlat6amBRSIRZ86cyQcPHvDIkSNMTEzkkydPGBISovIkbGhoyNWrV/Pq1ass
LS3l3r17VVoeRCIRXVxc5MpWEARaW1vT3d1drSy+u8Hq0qULW7durTaOZf36
9Zw1a5ZGJ1/g7eltwYIFzMjI4N27d1lSUsLg4OBmy3twcDCPHz+ussyuXbt4
+fJlTps2jTt27GBhYSEPHDhAfX19lfV0dHS4adOmZn0X8PYk12AOVZX3SBG3
adOGp0+fZtu2bdWW1dfXp6Ojo8I5Gx4ezp9++knh2A4fPpzR0dEMCQmhnp6e
3Er07sbsn1Y6/ww7Ojpy586d3LNnD2NjYzlr1ix6eHioVV56enrNOlG5ubnx
5s2bDAsL09huKhaLaWlpSTc3N7q5udHc3LxZwvFn2cDAQK15p1WrVty4cSPn
zZunsSnzr2ZjY2O2atWq2X2mq6tLBweHZtm3/2oWBIGLFy/mixcv5ImuVLGD
gwOXLl3KLVu2aHQq/WdYJpPRz8+P8+fPZ2RkJHv37q3RZlAqlbJdu3Z0dXVt
1mng380ikYi2trbN3tCamppy3bp1jI2NpZ+f35+au4sXL+bcuXNVlmnfvj23
bt3KuLg4xsTEcOzYsbSxsVH72+3ateOSJUtoZmbWrHfy8fFheXk5161b16xv
EgSBu3fv5v79+5vdl39ka2trpaZ1TViZXvn/FeBnTEwMtLW18dVXX2kMIf43
/U3KSCQSYerUqWjdujWWL1+usWv33/Q/iywsLPD7778r9V78T5BUKkX//v2R
np7ebJR9Hx8fFBUV4d69e/9RlPR/Kcq0VCpFdXX1fzvYdzMzM7x+/fpvhfM3
/U1/0/9KkkqlqKmp+cvSLqgiZUqn2ahzNjY22LhxI8zMzP7Ui+jq6v7bwDUb
Uq/+Tc0nCwsLTJkyBeHh4RgzZkyzx1dLSwvGxsbyOAF1Ser+EyQSidCyZUtM
nDgRTk5OGpUH3kax/5WAsM0lqVQKExOTZoNO/pl2mpMo7c+Sjo4ODAwMmgXc
KZFIYGJiAhMTk//WY/XvJFNTU+zfv/9Pg/b+VdRsCfrkk0/QsWPHP4WSqqWl
hUWLFiEzMxO7du1SWk4sFsPFxQWffvqpHHcoNTUVV65cUZtDviEPeUMUuSaZ
8KRSKQwNDeHk5ARbW1u89957AIAnT55g+/btaoNTgbeBh927d9cIE+yvJJlM
hlatWslTRCs6nerq6mLhwoUYMGAA9u3bBxsbGwwaNAhhYWEapRoWi8UIDg7G
3LlzUV1dDYlEgjVr1mDFihUKy1tZWcHFxQU5OTkoLi7WOKGYWCyGqakp7O3t
IRKJcOPGDY0zuOrp6WHs2LEYNGgQbGxsIBKJlKZr1tLSQmBgINzd3VFRUQEn
Jyc8ePAAc+fO1agt4C1cj6WlJY4fP67Rd1lZWaFt27b46KOPIBaLcebMGVy/
fl1tXUNDQ6xduxa9e/dG37591QY5/lkSBAFTpkzBr7/++qeyRWpKpqamCA8P
R58+fbBw4UKF2Gl/JDMzM6xcuRK+vr4gif3792PevHkKA3MFQYCLiwt69eqF
iooK/P7770hKSpLDd6kiXV1d+dpSX1+PyspKjdPV/zPUoOzVbahbt26Nzp07
a4xS3kCtWrVCnz5RJJQaAAAgAElEQVR9kJSUhJycHGhpaan9JplMBmdnZ7Ru
3RouLi4QiUTYu3evRutFs2Fwxo0bhxs3bijE8lFHenp6cHZ2Vgv0+emnn2LF
ihV48uQJnj59CkEQMGzYMBQVFWHRokU4d+5ck4heQRDg7OyMqVOn4qOPPoJM
JkN0dDRWr16tFDXBysoKwcHB8PDwgJOTE7S0tPDw4UNUVlbi3r17qK2tVbmz
a9u2Ldq2bYtjx47B1tYWI0aMUBkVrq+vjw8++ACdOnWCnZ0dAODmzZvYtWtX
k3fs1q0bAgICIBaLcfDgQTx8+FCeFlqdUtPT04OVlRV69+6N/v37o0OHDti/
fz9mzJihUAlra2ujffv2iI+Px4wZM6ClpYX4+HhERkZizJgxahd2qVQKOzs7
bNiwAXfv3sXmzZuVYuYJgoChQ4fi008/RZs2bXD79m2sWbMGly5dUrm4mJqa
YsaMGRgwYIAc1sfGxgb79+9X+W7A241IeHg4AgICsHbtWpw5c0YlaKWJiQkW
LFgAQRDkyACnT5+GIAiQyWRqbf9isRgRERG4f/++WqXTsmVLjB8/Hi1btsTN
mzeRmZmJoUOHwtHREePGjVNrwv7ggw8QFBSkkUlFJBJBX19ffgqtqqrC77//
rpGZvAH/79+5obK0tERcXBxqamoQHx+PWbNmAQAOHz6s9B1FIhF69+4NPz8/
pKenQywWIygoCDt27EB6enqT8jKZDGvWrEHXrl3lGYwPHTqE9evXIyUlRem7
GRkZYe3atfD394dYLEZVVRWys7Pl6d0vXLigcJ1p27atfBF//vw5KisrNdrE
NpC5uTnmzZuH9PR0bN++XWVZIyMjlJSUqN2Yv0v6+vpYu3Yt8vLycPDgQUgk
Enz77bc4fvy4yg3MlClT8PXXX+PAgQPIz8/HZ599hvfffx9jxoxRjzXYHO81
Nzc3FhUV0d/fX+4p8ccyqrhnz568cuWKyvgNc3NzJiQkcNOmTTQyMqK2tja1
tbXZq1cvpqSksKCggJ6enk3qubm5MS0tjVFRUXRzc+Po0aOZkJCg0iW7X79+
rKys5J49e+jr60s7Ozvq6enR2dmZCxYs4MmTJ2lvb6+wbuvWrXnmzBl59PiC
BQs4f/58hWUbPIZWr17Nly9fsrCwkDk5OaytrWVGRoZCL6LDhw/zzZs3rKio
YHV1NZ8+fcpr164xJSWFUVFRSiHfTU1NGR8fz/v37/PcuXOMjIzkoEGDVEb9
6+jocMKECY0QC4YMGcLHjx9rFDcCvPUCtLa25tq1a3nv3j2FUOcNbGhoSJlM
Rg8PDyYnJ/PWrVtqPYH8/f2ZmprKSZMmUV9fn/369dMYUWDIkCG8fv260gDc
P7KdnR3v3LnDsLAwSiQS6ujoUCqVcsSIEZw1a5Zal9zWrVszPT1dJQJEw/wJ
Dw/nN998Q11dXQqCwHbt2jE5OZmxsbFq55dMJuPOnTtZU1PD/fv3K0Qk0NbW
po+PDydPnszvv/+e8+fP57x58xgZGcl58+YxPDycXl5eaj2kdHV1GRsbq9QN
XFtbm25ubgwNDeWkSZPo4+PDvn370t3dnb179+bEiRM5bNgwlS7G3bt3Z25u
Lj08PCgSiThw4EBeunRJpQeroaEh09PTmZCQQAsLC44aNYqHDh1ihw4dOGjQ
oCZjJZVKuW7dOj5//px1dXWsr69nbW2tytAMQRDYt29fvn79mnV1dayoqOCl
S5eYnZ3N169fs6SkhMOHD28yXoIg8IsvvuDixYuZkJDA06dPc/PmzSpjnN5l
BwcHxsbG8sGDBxw8eLDa8uPGjWNmZibNzc0pEoloYWFBS0tLpV5sIpGI06dP
Z3x8vDy5oYmJCa9cuUI/Pz+VbbVt25ahoaFs164dfX19effuXU6aNKlRH/xL
XKa/+eYbPnjwgK6urpwwYQKjo6M5duxYtbkogLeL0pw5cxgaGqpyMo0dO5bF
xcXs3bt3k785OzszLy9PYYesWLGiURSyn58fd+3apTKeoE2bNkxKSuKpU6fY
tWtX2tvbMzIyklevXmV0dLRc+P9Yz9TUlAcPHuTu3btpYmIiV3guLi4K2/Hy
8uKFCxeYl5fHjRs30sPDg3369GFFRQUzMzMVKh1HR0cGBATQ39+f06dP5+DB
gzl48GCOHDmSmZmZSgNLfX19mZeXx549e1Imk1FXV1cjl8s/jomjoyMLCwuV
KtJ3WVdXl19//TWvXLnCW7du0c/PT+2CKRKJ2LNnT969e5dbtmxRG/ehra3d
aMEaN24chw8frvbdrKysmJCQ0ChuS12MjyAInDJlCh8+fMgOHTrQzs6Os2bN
4rp169ixY0e13xYUFMSMjAy18VuCIDAsLIzJycns1auXPDhy5syZGrnYDh48
mC9fvmRBQYFS2JiGKH1vb29aW1vL87qIRCKKxWKamppy2rRpavMotWzZkkeP
HlXqZu3p6cmdO3dyxowZHDduHG/fvs28vDzu2bOHCxcu5IQJE3jnzh2VcVnL
ly/n1q1b5f0rkUi4d+9eTpgwQWkdsVjM6OhoFhQUsGPHjmzfvj03bNjArKws
3rhxg9bW1k3qSKVSBgcHs6CgQK50QkNDlbahpaXF0aNH88WLF6yrq2NCQgIN
DAxoa2vLTZs2sba2lkuXLlW4VgiCQJFIxNGjR7Nfv34MCAjgmjVrVG5+tLW1
6efnx5MnT3LSpEmMjo5WCOnzx37Yvn27fKzXrl3LzMxM3r17l+vWrVMo8506
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
PXv28Nq1a7x+/Tpzc3NZUVHBoqIi5uTkMCEhgSNHjmxisrCzs2N+fj4vXLhA
FxcXikQi2tjYyGFnlJkd9fX1uXPnTv7yyy8cOXIkDxw4oFZ2G9jJyYkFBQUa
2+3btm3L3bt309zcnE5OTkxMTNQo0n3t2rV88+YNr127ptY0aWZmxujoaE6d
OpWDBw+WowQfPHiQ/fr1o4ODg8q7T5FIxGXLlqlMo62M9fT0uH37dm7YsEFl
hLyXlxeTk5PlpnpBEDho0CDeunVLKT7c5MmTmZ2dzTdv3vDNmzfcuXMnnZyc
1M7dzp078969e43uhq2trZmZmakQjRkAIyIiWFdXx1WrVsnnQFhYGCsrK7l4
8eImGHu6urrs27cvo6OjmZGRwXv37jElJYU7duzgrl27uHHjRo4ePbrJ/BcE
gWZmZnIzpqWlJY8fP64WiV0mkzEpKYn19fUsKCjg6dOnmZ+fz5ycHIV34O/O
fWNjY3p7e3PatGnMz89XeRdpZ2fHbdu2MTAwUP7uPj4+zM7ObiK3/7R5LTU1
FXl5ebCyssK8efMwceJEWFhYAIDKXa1IJMLo0aNx7949te6cr1+/xrFjx/D5
55/DxcUFmZmZ8PHxwebNm2FoaIjw8HBcu3ZNYd1Xr141SuXq6uqK0tJSldkr
P/74YwwcOBDLly/HuXPn0KJFC+Tl5cHW1lbtya2mpgbt2rVDQkICzpw5o/K7
3qU2bdpgwoQJIImTJ0+q9WSprKzEmTNncObMGbi6umLYsGH4+uuvG50C3qXq
6mocPXoUXl5ejZLf1dTUICUlBVFRUZgyZQq2bdsm/7tYLIa1tTW0tLRw8uRJ
DBgwAAYGBqipqcGBAwdw4MAB/Pjjj/K+9Pf3x9ixYxESEoLS0lKYm5sjNDQU
n3/+OWbPnq0y3S35Nr9HdXU1CgoKkJubizt37uDWrVsoLi5GeXl5k353d3dH
bW0tUlJS8OTJExQVFcHBwQGZmZlIT09HUVGRwtNBdXU1SkpKoKOjg549e+KT
Tz7B4MGDkZ+fjwULFiiVDWdnZ/To0QPDhw/HxYsXQRKRkZEYOnSo0twp71J5
eblGyawAwMHBASKRCFVVVcjJyUFWVhZcXFxU5pIB/t/OtsHd+rffflP6PTKZ
DC9fvkRGRgYsLS3x0UcfIS8vD1u3btXI00kkEkEikTTLKwp4awKcN28eHB0d
MWbMGJWmSZLQ1dWFnp6ePMHY7NmzsX//foX5YExMTDBx4kT8n//zf3Dx4kV0
6NABpaWlyM7OVmvafe+99/DLL7/g1q1bchNyp06dIBKJ1CaMvHfvHrS1tTF4
8GCEh4cjLS0N69evb3ICtLGxQefOnXH69GksXboUlZWVqK6uRl1dHcRiMczN
zdGtWzdYWVk1msskG/WzjY0NpFKp2hThVVVV2LRpE9zc3GBjYwMbGxtkZmbi
m2++QVpamtJ6tbW1KCsrQ3p6Or744gskJCQoPVkCwKNHjzB37lx88MEHkEgk
EIlECA4ORkpKikZpzIFmmNeuXbuGXr164bPPPoOjoyMEQUBycjLi4uJU5vv2
8PDAkCFDMGHCBLX28Pr6emzZsgU2NjaIiYlBYmIiBg8ejFevXuHLL7/EpUuX
NPaJf/PmjdogRSMjI7x+/RopKSno1q0bBgwYAHt7e/zwww9q7dxubm7o0aMH
vvzyS41igYC3C8XYsWPh4eGBU6dOqczJrqhu3759cfLkSbUmr8OHD8PAwEBh
kq379+/DwsICurq68glWXV2NBQsW4Pnz5xgwYACqqqowefJkpKenw8/Pr1G6
X+Ctff+jjz5CfHw8SktLYWlpidraWsyaNQs//fSTSkX/6tUrBAcHQxAEvHr1
SiN33YMHD+LQoUONymppackzeyqjJ0+eYOrUqfjyyy8xfvx4ZGRkYPPmzTh2
7JhKM4AgCNDT08OAAQPw22+/oVWrVrC0tNTofsva2hr6+vpqZb2BsrOzoaur
i+XLl+PAgQOoqanRyDbekI7YwcEBY8aMwdKlS5X2e0FBAVatWgU7OzvcvXsX
x44daxbkS11dHZ4/fw4jIyON64hEIkyYMAE9e/bEyJEjVd4RAG/lsri4GHFx
cZDJZLCwsMB//dd/NdrsvEtGRkbQ0tICSXk81ZUrVzRKkf748WP4+vpi6tSp
MDAwgIODA7p164ZVq1YhOztbYZ0GOfP09ISxsTG+/fZbPHr0CNOnT0dhYWGT
8jk5OZg3b57Sd2hIlqaOXF1d8fDhQ7XyRBIHDx7EkydP5G7de/fuRWpqqto2
gLdhKl27dkXv3r1Vmvt1dHTwj3/8A15eXpg4cSK8vLzw4YcfYuzYsRqvg/92
wE9vb2+NXDLfZT09PW7atInFxcU8depUsyDBG9jNzY3Xrl2jubm50jLOzs68
dOkS7969ywcPHvDcuXNyN1lVv62rq8stW7Zw4sSJzXIbt7Cw4M2bN/nkyRON
zTUN3LVrV16/fl0tcrE6trW15fjx4xWa9Rq8mdQhRUulUjo6OrJfv37s378/
+/Xrp5FJ6D/FEomEpqamGgMgtmjRglOnTmVkZCSnT5/O+fPnMzAwUKOxdnd3
5/Hjx5sF8GhmZsbZs2fz5MmTnDhxokYI2q6urly8eDFnzJjxlwB3duzYsVnz
0MzMjElJSRw4cKDGdezt7Tlv3jxOmzaNTk5OKtNCCIJAZ2dnhoaGctCgQezW
rZtGKOzAW1PUhg0bWFBQwOzsbCYlJXH27Nkqx6xz584sKSmR52U6ePAgnZ2d
mx020lyePHmyWvflf5a1tLS4bt067tu3Ty1qtlgsZmhoqDyvVVxcnFJv1f8o
yvSfYQMDA5qbm6uM6VHFdnZ27Nu3r9q4jIZ2zMzMqKenp1EchyAINDY2bjaE
ukgkYlhYGBctWtTsbKh9+/blsmXLmpUE72/+61ksFquduMrq6enp/SUI538F
29nZMSIiQmNF8FezTCajmZkZjY2N2aJFC7XKQ1tbW56OxM/P7y+bh126dPlL
EOYNDAzU3i+/2xcNfSeVSpX23f8KlOn/CSQWiyESiTQ/iv5/JBKJIAiCRuaD
v+lv+k+TIAgQiUR/y+v/YvqXAX7+Tf8c1dXVNVvh9O3bFz169PhvN4FFIhEs
LS3lcEAWFhZ/CpPvz5CrqyuCg4P/EgDK/wnk4+ODUaNG/WX9r45I/reT1/8t
ZGNjg8jISHh4ePynX0Uh/WUz1t7eHlOmTMHRo0eRlJSk9qJeEAR06tQJY8aM
gaGhIX799Vds2rSpyaV2A2lpacHGxgYSiQSvX79uFBfzP5l0dHQwc+ZMXLx4
EYmJiRrXEwQBNjY2IKnWE+pd0tbWhrGxMYyNjSGVSvHkyROFsRV6enr4+uuv
4e7uDpIoKyuDhYUFLl26hDVr1mh8kQ4A/fr1g7m5OXbs2KExDtiIESMgEok0
diyxsbGBp6cnysvL8fLlS7Vgmtra2nLg14KCAo281v5T5OrqiuXLl2PNmjUa
Q9qbmZnB0tISOTk5ah0XxGIxDA0NYWBgIAdo1WSc9PX1MWfOHBQUFODHH3/U
6L2Atzh73bt3h7W1NWJiYjQGddWUGuJ7ZDIZysrK8Pz582ZvBP+7kqmpKTZu
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
/gurV6/Gb7/91mRsHz16JE9xfvr0aZSVleHp06ewsbGR5z2ytrZGRkZGEzNv
165dUVdXh08++QRffvklcnJycPPmTXh6eiIlJUUptI+enh5mzJiBy5cvN4I0
UhXknpWVhe+++07et2KxGB4eHjA2NlZZr6ioCL6+voiJiUFpaak8DcuePXvU
4iECgMqTjo6ODtu3b08LCwuKxWIGBQVx+fLl7NixIy0tLWlmZsZ9+/Zx/Pjx
SrVpcHBwE7iXLl268OjRoyqP6H369GFVVRUPHTrEqKgoZmVl8cWLF/zpp5+U
Hh2lUil37drFN2/e8MWLF8zLy2P37t2VtiOTyRgXF8c3b97wwoULcmRVBwcH
FhUVMSUlReVJpGXLlmzbti19fHy4e/dulfEVxsbGjI6O5v379zlz5kzu2LGD
J06cUOumGBoayjt37vD7779nSUkJq6urWVNTw4qKCnbt2rVJeTMzM2ZnZ9PX
11fulm1nZ6cS4qOBdXR06OLiwm3btsnhPZrrwuvl5cXjx49rVK9FixY8cuQI
T58+rVEMTYsWLXj06FHm5+dzy5YtPHHiBMeMGaOyrS5dunD+/PkcPHiwxnEm
AQEBrKqq4oULF6inp0dBELhr1y7euHFDqbmsc+fOPH/+PHV1dSmRSDhlyhQW
FhayZ8+eStvR0tLiL7/8wilTptDR0ZGdO3fmsmXLWFhYyBkzZiitJxKJGBQU
xLy8PPr7+9PGxoZubm4a9eGkSZO4fv16rl27lqtXr9bITbYBldrc3Jy3bt1i
YmKiSlOeTCbjkSNHGBMTw/j4eCYlJaltQ1dXl2fPnuXKlSvlu+wxY8YwOTlZ
5clUS0uLq1atYlFRES9cuMAuXbrQysqKrVq1oq2trcI+MTc357p163jp0iVW
VVUxISFBIRr1u2xtbc3U1FQeOXJEo7jDDh068NChQywpKWFJSQkfPXrElJQU
5uXlsaysTKFpXUtLiytXruSgQYMokUjo6OhIb29vbtiwgS9fvmTnzp2Vtmdn
Z8cHDx4wICCAwNtTX0hICKOjo9Wm2Gjgrl27MjMzU6352cjIiD/99BNTU1P5
zTffcM+ePezTp0+TMJB/SZxOg+2+4d8jRoxgQkKC0qAqLS0tnjx5ssnRNTAw
kHFxcUqVgUgkoru7OwsKClhTU8M3b96wqKiIc+bMUWlG8PHxYX5+PiMjI9m+
fXvu3LmTeXl5Su3j1tbWzM7OZk5ODh0cHOTf17NnT5aWlnLWrFlKj5lisZiW
lpYMDg5mXFycSoEAwIULF8rxwlJSUpifn98oFYMyjoiIkCuZgwcPcuDAgRw1
ahQLCwsVfpe+vj6PHDnCzMxM7t69m4GBgQwLC+PNmzdVmsnatWvH2NhYZmVl
MSYmhuvWrePly5dVYtcBb23wISEhnDRpEt3c3DhgwADu27dPI5u/q6sri4uL
NTahfPHFF6yqqmJtbS1zc3OZlJTE69evq5QJsVjM2bNn8/Dhwxw5cqRGcSNO
Tk4sKSlhQkKCfNFatGgRb968qVTp+Pj4MC0tjU5OTgwJCeGLFy+YnJxMIyMj
lW316dOHSUlJcmzBx48fc/DgwSo3O66urrx79y6nTJlCd3d3ZmRksLi4mGFh
YWrvdYYOHcpLly5pHPPVoUMHrlu3jqtWrWJYWBifP3/OcePGqTRFicViurm5
0dDQkNHR0Tx37pzadrp06cKcnBy6urrKnwUEBPD8+fMqlY4gCLSzs6OdnR1H
jRqlUd4ZGxsb7t+/n1lZWXz58qVS6P93eeLEibx27RpbtmypkawCbzdJ9vb2
7NatG3v27MkJEybwxYsXXLJkicI4vYZ8Sn8c+7Fjx/L69es0NjZW2pa9vT1z
c3Pp4+NDMzMz7ty5kzk5Obx27RqvXr2qNli5VatWTElJ4bJlyzS6u2zXrh3v
3LnDiooKDhgwQOF8/5cHh3bo0IEXLlxQuNtuYIlEwvT09Ea5OvT19RkfH6/U
1m1ubs5Zs2bJgfyePHnClStX0sXFRWVApa6uLk+fPs3Y2Fi5AFlbW/PgwYOc
OXOmwkni5ubG0tJS3r59W37306pVK6alpTE/P7/RBGhgmUxGT09Prlu3jpmZ
mSwuLuaOHTs4f/58jhs3jh4eHgoFuGPHjvT19aW9vT1NTU25YcMGfvHFF2oH
d+bMmayvr2d5eTmnT5/OVatW8f79+9y9e7fSSHQjIyN6enpyxowZ3LZtmzxn
jbKFQiqVcseOHUxMTGRAQAAlEgkNDQ15584dlRHlEomEK1eu5M6dOzlv3jxe
unSJBQUFjImJoZmZmcqdt46ODjdu3MiMjAyVSbre5a1bt7K+vp65ubl0d3eX
5/BRpXTMzMy4cOFCmpubs2/fvlywYIFKu/i7/VFQUEB3d3fq6OgwLi6Oqamp
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
UKC2f5MnTyZbW1vq7OwkPp9PN27coJiYGGprayMLCwvy9PQkhULxVj//8Y9/
UEFBAW3cuJE2btxIVVVV1N7eTrq6uvTBBx9QREQEpaenv4MUbGlpSfHx8eTo
6EhNTU3k6upK3377Ld25c4esrKxILpfTjBkzWFFx7e3tydbWlqKiolhG/X9I
KBRSv379aOnSpfS73/2O1qxZQzdv3lRbvrOzk7Zu3ar6e0BAAAUEBHRb1sDA
gD755BMSCoX0+PFj2rdvH+3du5cmTZpEvXv3ppqaGvr222+7RRPX09OjGTNm
0N/+9jc6cOAAWVtbq9B71RGPx6PAwEAyMDCgb775hm7dukWTJ0+msWPHUkdH
B92+fVvtfnF1daVXr17RwIEDqXfv3lRRUaG2HSKi9vZ22rRpE8nlcgoNDaVd
u3Zp3MMSiYRiY2Pp97//Pa1bt44++ugjGjJkCJWVldHjx49p0KBBlJGRQU+f
Pn2n7sOHD+nhw4ckl8tZ9+DPqbi4mL7//nuKi4tjRabm8Xg0ffp0GjFiBCUn
J9OdO3eos7NTFavz6tUrcnJyIpFIRD/99JOqnqmpKW3cuJHCwsJIIBCQs7Mz
ffbZZ2RpaUnLli1Ti/zM5/NJLpcTwzD0/Plz6urqIoZh6L/+67/IxsaG0tPT
Wc82CwsLsrW1pbVr13Iah0ePHtH169fJ1taWdHR0iMfjUW1trdryz58/pwMH
DtBnn31GpaWlVF5eTq9evSI9PT3i8Xhq96Curi55enrSs2fPaPXq1d3OJRv1
SOhMmjSJysvL6fvvv1d7UL5JJiYmZGBgQOfPn6ehQ4eqhcxW0tWrV+nq1as9
6ZKKUlNTycjIiKZPn0729vZUVlZG8+fPp6ysrG4nViQSUa9evSg+Pp5MTEwo
NDSUampqSC6Xq22DYRgKCgqiCRMm0PHjx2ncuHHk7e1NmzdvZl08z549o4aG
Blq+fDk9efKEsrKyKDY2lm7dusX6TRcuXKCoqCgKDg6mkJAQYhiGHj58SOfP
n6fExMR3oHfMzMxUEO9XrlwhIyMjGjp0KDU2NrLmqE9LS1MJuCtXrtCxY8fo
1q1bqjnes2fPm69fIiJ69eoV/e1vf6MFCxbQ3//+d2ptbaXOzk7S0tIiIyMj
+u///m86fvz4O21ZWFiQo6MjbdmyhVJSUigiIoJGjx5NHh4edP36dfr888/p
5MmTrOvL2dlZlcZCHRkYGJBQKKQ//elPNGnSJLKxsaHc3Fz661//SnV1dRrX
r76+PjEMQ46OjhQeHk5btmzptlxzczOFh4eTo6Mjfffdd3TlyhX66aefOB0U
P/30E7W0tBCfzycXFxdaunQpNTY2dguxryRjY2Py9fWlJ0+e0LZt26impoaO
Hj1Ky5cvp169etGXX36p9hDs1asXubq6kqOjI8XFxb2VcoSN3N3dqauriwoL
CzWWdXBwIF9fXwJA8+bNIwAklUpp2bJl1NDQQImJiZzb7QmJRCLq3bt3tylT
lNTZ2Un79++n+Ph4mjp1KpWXl9P9+/fpD3/4A9na2tLDhw9p//79bwkcoteX
4fb2dtqwYQN1dHQQn8+n48eP07Jly1jPtLFjx1JsbCzxeDyqq6uju3fv0qtX
r8jPz49ycnIoPT2d9ZsGDhxIV69e5QTJRfR6Pf3tb3+j+Ph4EgqFFBMTQ//4
xz9Y66SmppKtrS3t2bOHSkpKqKysjAICAigvL09tu56enmRvb095eXlUXFzM
SRa8ST2CwRkwYAC5urrSwYMHOQ2Erq4ubd26ldzd3amkpIQWLlyoMb/IP0MM
w1CvXr1IT0+Pmpub6dmzZ2oHxNrammbOnEkPHjygI0eOsB5gb5KLiwt9/PHH
ZGxsTB0dHXTy5Ek6e/Ys682ciMjc3JxkMhk9f/6cmpqaOGO1MQxDenp6pKOj
Q0KhkJ4+fUoPHz7s9rvkcjmFhISQvb09vXr1ip4+fUrZ2dmqVwgb8fl8AsDp
VfkmCQQCcnNzo8GDB1OfPn2osbGRSkpKqLKysts8LxKJhExNTen+/fvU0dFB
YrGYTE1NSSgU0pMnT+jRo0caF/Hy5cuJYRhat26d2jIRERE0atQoampqouPH
j9PZs2fp3r17nDeIi4sLrVmzhtra2ig9PZ2+/fZbjXP8PjR69GhatmwZGRoa
0n//93/TihUrWC8Irq6uNGPGDCotLaXU1FTVfAmFQuLz+azz3KdPHwoICKCC
ggKqqKjgPBZubm4kkUjo4sWLGsvKZDIyNzenjo4O+vHHH1+rUwQC0tHRoY6O
Dnrw4IHGtS+Xy+m7776juro6CgoKYs21pCSRSESrVq2itLQ01hccwzBkYmJC
EydOpL59+6qi9tvb2ykpKYkqKyvf2QMMw5CRkRHp6OgQAHry5Am1tLRo3Ct+
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
TZo0iSorK6lz585K7WlaWlq0Zs0aKioqol27dpGLiwvr9iTriY1HnyyeOHEi
VVdX0+bNm1uprRmGobZt29K0adNowoQJUq9GNt9IJBJJVadsYss4HA55eXnR
tGnTaPr06eTv7092dnZK7SYfq+GVcZs2bej69euUlpZGS5cupV69erG2W9ra
2lJGRgaNGTOGVZumpqaUmZlJz58/p4kTJ7K2q0o4MDCQLl++rDRuSFNTk44e
PUrXrl2jCRMmfOC5p62tTebm5nKdP8zMzGjZsmUUExNDcXFxNGrUKJlz/h+B
vebh4QFbW1scPXqUNcSDkZER9PX1UVxcrLQsj8eDQCD4IF6msbHxL0UH5nA4
sLa2RseOHWFkZAQiQllZGfLz85WqkzgcDnbt2oXY2FhcuHDhL+uTIhIKheDx
eGhpaUFDQ4PC2xYbpOL3SSQSoVevXiAiXL9+HZWVlSrPtZ2dHX7++WfMnDlT
ZYTrfydxOBwMHz4choaG2Lx5s0y1DZ/PBxGhsbFRpXwwEmrbti327NmDyspK
REdHIyUlReF88ng8CIXCv3yNKyMNDQ384x//wKNHj5Tm5PlvoPfVeI2NjSqt
U4ZhoKamhsbGRlaIz1paWvj+++/xxx9/4MyZMyq/ks3MzKCjo4OSkhKl/VRT
UwPDMErPBVkkOWtbWlrw5s0bmfXlYa99Bvz8NxKfz0eXLl1w9erVv0Xl8r+R
GIYBl8v9pEP6300Mw4DD4bBOS6EqcTgcODg4wMzMDBcvXlQaQP2ZPtN/Mn0W
Op/pM32mz/SZWpG6ujosLCxw9+7dv1Tb8JehTIvFYkyePBnHjh3DmDFjWEPN
qEIGBgaYPXs25s6dywpy468iIyMjeHh4wN3dHdra2qzriUQidOrU6T8S6dbK
ygoeHh4Qi8WsEap5PB46deoEsVisUlvq6urQ19f/jIT9byAulwsbGxt4eXmh
W7duCr0S/yqysrJC586dWe8NCeyRs7MzrKysFK4LDocDR0dHmJmZsQbP/JiM
jIw+CVzV0tISK1euZAVe/GeIy+XC0dERFhYWKtVTU1PD9OnTpdmD5ZGGhgZm
zpyJbdu2wc7OjtVv8/l8LF68GB4eHir1SUKS1NWWlpYwMzNjV0nV4NBjx45R
QUEBPXjwgGpra2nKlCmsDJgcDoc8PDxo4cKF5OXlpdBQ6OzsTGvXrqU1a9ZQ
TEyMQqMdn88nLy8v6tOnD1laWqpkIHyfbWxspPEFDQ0NlJqaSu7u7qzqdu7c
mR4+fKgwtuf9eRAKhSonm1JlXAzDkJ2dHYWGhtKtW7eovr6eiouLFQb0SVgg
ENCcOXNow4YNlJKSwioAWCwW0/z58yktLY0uXbpEMTExrCBFgHdGbbFYTI6O
jmRoaEgcDofc3d2VGpLZzoeenh65u7vT3LlzaePGjay+0ftsaGhIO3fuJGdn
Z1bf1s7O7oP9oKenJ3N/8Pl8sre3p8GDB9O6detox44dFB4erhRBQ/KbK1as
oDt37tDjx4+psLCQTp48yXrOJfNuaWlJXbp0ITc3N6XIIi4uLpSamkqnT5+m
CRMmkL6+vlIjd1BQEGVlZVFqaiplZ2crBPy0sLCg27dvU0lJCZ0+fZpCQ0NZ
O81I+Mcff6QtW7aoZHwXi8WUnJwsjSlSpT1V9qS2tjbNmTOHnjx5QikpKayQ
SIB3wbGbNm2i2tpaOnXqlMLYPh0dHQoODqZz587RsWPHpFBTitjV1ZUyMzNV
Wjvvr+GIiAhKT0+nI0eO0LJlyz5wPvjLEAns7e3JxMSEJkyYQLW1tRQeHq70
I1tZWVFoaCiVl5fTpUuXKCcnh6ZPn670YLG1taXk5GSFE21oaEgXLlygQ4cO
UVJSEi1ZsoQcHR1V9g5btGgRrVu3jnx9fSkrK4vGjRvHCuuMYRhatGgR3bx5
UyZOmVAopN69e9P48eNp4cKFlJiYSOnp6XT8+HGaMWMG2dvbK50/CwsLio2N
/QB2wtramiZPntxqsZiamtLo0aPp1q1bVFtbKw3Ii4qKaoW/JIsHDhxIq1at
Ii0tLdq4cSMtXLhQ4aWiffv2dP36daqvr6e5c+eSl5cXXb9+XS6ysoQ5HA65
uLjQunXr6MqVK3Ty5Ek6dOgQubm50erVq1t5phkYGNCoUaNo7NixtGLFCtq5
c6dSDx2BQEDbtm2j6upqevz4MT18+JCSk5NVigp3d3en7OxsVvAlFhYWtHnz
Zunvt2/fnpYtWybz0hQcHEyzZs2ioKAg6tChA5mbm5Onpydt2bJFKTp6YGAg
VVVVUWVlJYWFhZFYLKbr16/T0KFDWa1XJycnSkhIoLKyMnrx4gU9efJEIYYi
l8ulZcuWUZ8+fUhdXZ20tbVpxowZZGRkpLCdn376iTp37kx6enqUkJBAEyZM
kFteKBTSgAEDaO3atXTq1Cl6+PAh3bx5UyVss6CgIEpKSmK9942MjGjFihU0
aNAgpd5oDMOQtrY2OTs709ChQ2nNmjWUlZVFoaGhrIScl5cXlZeXU1NTE9XV
1bFCwDcwMKA9e/bQixcvaP78+dS2bVtWKP22trZUUlKidA8yDEMLFiyg1atX
f5JXp7+/P508eZKMjY1lZtn9S7HX7D2pkUgAACAASURBVO3tKT8/n+7cuSPF
mZI3KBcXFzp37hxVVlZSaGgo6enpkaOjI127dk3hzUdTU5M2b95M+/btU/pR
DQwMpGB148aNo9jYWFq7di11796dVdQwn8+n1atXk5OTEy1btkyahoHNxLdt
25aKiopo1KhRMv+ur69PcXFxdOLECVq6dCkFBQVRUFAQhYWF0Z49eygvL482
bdokxd/6mM3MzGjbtm306NEjqbuunp4eJSYm0v379z+A++BwOLRkyRKqqamR
wrK0b9+eANDChQvp1KlTtH379g+gWt5nkUhEu3btkh7mnTt3psTERLlzyOFw
KC4ujl68eEErV66UulqvXLmS4uLiFK6LYcOGUVlZGcXHx5OTkxMZGBhQYmIi
5ebmyoTdt7S0pJkzZ9KUKVMoICCANm3apBCqCHh32cnKyqL09HRyc3OjdevW
UWVlpdzxy+LAwEC6c+cOK+j6kSNH0rp166Rratu2bXJRArS1tWXelDt06EBr
165VuP4ksC07d+4kExMTYhiGUlNTadq0aQr7JxKJKCgoiMrKyujy5cu0fPly
2rx5M9XU1ChEQVBXV6f4+HjS0dEhdXV1Wrp0KSt3Xi6XSwKBgHx9fenGjRus
MfnU1NSoQ4cOlJmZyQqdAXgntOLj42nRokWsXiBcLpcmTZrESpshEAho0qRJ
dOrUKaqqqqL6+nopNNCRI0eUXmIsLCwoIyODampq6MCBA1RfX690XLq6urRn
zx6qqqqiCRMmqJSehMPh0LZt2ygiIkJhOW1tbcrOzv4AV1BbW5uMjIyUCm6h
UEgJCQkKz/C/ROhwuVwaMGAAFRYW0u+//04+Pj5yPzDDMNSzZ0/KzMyksLAw
8vX1lW52iYRNSEiQ2+Hhw4dTeXk5ZWZm0oABA1ghRr+/STp16kQLFy6kNWvW
KF0UEtDA9PR0SktLU5orRMI8Ho8iIyPpyJEjSvO5aGhotJorgUAgzVMkC5bF
zs6OkpOT6fHjxzRlyhQSCoXE4XBo0aJF9OjRI/L39291OLm5udGgQYPIysqK
DA0NicfjEYfDof3799ORI0fo/PnzUkH0Mfv4+HzwsrG1taW9e/fKHRvDMDR1
6lQKCAj4YFMsXbqU0tLS5K4NHR0dysrKooiICGkO+enTp9OdO3do+vTpShc8
j8ejrKwsuYL+/c1nZGQkxZFKSkqi0tJS1t9XQ0ODjhw5Qrt27WJ1mx03bpz0
MHFycmINdfR+f93c3D5A75VX7n1sLDs7O7pz5w6NHj1abh0jIyOKjo6m0tJS
CgkJIbFYTIsWLaI7d+5QQkKCQlgrfX19OnPmDFlaWtKkSZNo7ty5StVxYrGY
Bg8eTJs2baKCggIKCgpSSe2lpqZGW7duZR1LZG1tTbdu3WINNePk5ESLFi1i
9Spq3749lZWV0ePHj+nKlSsUFxdHu3btopcvX1JUVJTC3xCLxVK1/fz582n0
6NH0/PlzhS9nPp9PkZGR9PTpUwoJCfmkfFjr169XKnSMjIzo5s2b5OLiQsA7
DMWTJ09Sfn6+0jxaOjo6dPjwYfrxxx+pd+/eMmPm/jQMDgAMGjQIO3bsAJ/P
x+jRo5GamirXv7tbt26IiIhAbGwstm/f/kG+GCLC69evFRr9UlJSkJeXBx0d
HXTv3h2+vr749ddf5UbUv0+vX7/GzZs30bVrV3z55ZdK89xoamrC0NAQbdu2
xbfffssqNwbwLnJ4wIABGD9+vEIXaFl/4/F4+OqrrzBt2jQUFRUhJyfng79b
WFhg3759cHV1RUVFBaytrTF79my8evUKQUFBSE5OxvHjx1t5m+Tl5QGANEMm
EcHJyQlff/01ZsyYgdLSUpnQMVwuF76+voiJifngNxW5BxMRNm7cKP1v4F1c
UPv27XHt2jW5a4PH40FTUxOXL19Gjx49MG3aNFRXV2P79u3YsWOHUvdpSUZZ
ZWgEwLu5FwgE0NHRgZaWFl6+fImXL1+Cw+FAQ0NDYbyYtbU1evXqhX/+85+s
3KQlsTocDgc9e/aEkZERjIyMFMZvCYVC9OjRA0ZGRmjXrh06deqE+fPny507
hmEgEomgoaGB5uZmcLlcTJ48Gc+ePZMLXaStrY21a9dCV1cXgYGB0NTUxObN
m9GzZ08cOHAAU6dOVYhAIUmdHB4ejocPH2LZsmUKI+QZhkGfPn3Qv39/6Xxn
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
tbW0Y8cOuR44Xbp0oYcPH9LgwYOJx+NRQUEBZWdnK0VzdnFxoezsbLp27Rq5
u7uz7pcEliYzM5Nqa2tpypQpSusEBQVRbW0tNTU10YsXL+Qu8o+Zx+PRunXr
KDIykgwNDcnd3V3pocnn82nTpk2Um5tLFhYWlJmZqVToTJo0iWxsbIjP57MW
biKR6INLjrm5Oa1bt47Va0xPT4/y8vKUoqJraGjQhg0bqKGhgXJzc6VQQ2pq
anT69Gm6fPmyzH2io6NDubm5tGDBAuJyuRQUFESpqamUlpZG8+fPV+rNxePx
aM2aNbR8+XKlYwkICKCVK1eSiYkJcblccnJyYp1rSiAQ0I4dOygxMfGTPPO4
XC5NnDiRbt68KTc3k6WlJR08eJDu3btH/fr1I4ZhpGvk7t275OTkJLPe2LFj
KT8/X6rVCAwMpBMnTiiEY+JwODRlyhRKS0ujyZMn0/3792n69OkKx6Curk57
9uyh1NRUsrOzo3Xr1tGGDRuUzp++vr70Rc7hcGjRokUKEabfX1OzZ8+mtLQ0
Wr58uULUe8k83b59my5cuEDbtm2j0NBQqqiooCdPniisa2xsTPHx8dLX6+jR
oykgIEDpNxWLxZSTk6PwhfiXCB09PT2KiYmhvLw8qqyspLCwMJVzbBQWFrJy
a3RycqJdu3bRTz/9RIcPHyZ7e3tWbUyfPp2qqqqk2GYDBgxQupjs7e3J3d2d
Vq9eTfv371dpTAEBAfTs2TPau3evwhtFWFgYZWdnk7a2NpmZmdHt27cpJiZG
4RNYAs0fGxtL5ubmKgl4DQ0NOnToENXV1dHVq1dZpQ5QU1Oj5cuXU1NTEzU1
NVF6ejqrC4KnpyelpaVRmzZtaPXq1ZSZmakwqRiXy6Xp06dTRUUFeXt7k42N
DZWXl9OiRYvk1hEIBLRo0SJSU1MjhmHIwcGB5s+fT4MGDZIrQMzMzCgpKYkO
Hz4sFVIWFhYUERHB6rB1dHSksrIyuaoxCTs7O1NpaSndvn2bXF1dP9iYt2/f
phkzZsgUWh4eHvTgwQPq2rUr6ejo0O3bt2nfvn00Y8YMKikpUYr3pqmpScnJ
yUrzzvD5fJo5c+YHAoNhGDI2NqaxY8cqXFcMw9CwYcOosLCQOnTowEplI4vt
7e3p5s2bMoWHra0tnT9/niorK2nw4MHSuXJ3d6eKigqaPXu2zO+lr69Ply9f
pqVLl0r/beHChVRUVKSwj7q6upSTk0MpKSl08uRJqqioUHq+TJ06le7fv09e
Xl6kqalJOTk5NH78eOk8sllPfD6ftm/frlTAAaDOnTvT5cuXqXfv3qzU3CKR
SHqhkCSxfPHiBaWlpSnciwzD0JAhQ2jv3r3Uvn172rBhAyuMPEdHR6VC909h
r0noxYsX2L59O8rKyhAVFQUfHx/07t2bVV1tbW18++23+P3331lFyRYUFOCH
H37A8ePHoampySrgk8/nw9vbGwKBAOfOncPly5dhbm6usM7r169RUlKCS5cu
4cKFC7CxsWEd1dynTx+sWLECdXV1WLt2rcKArD/++ANisRgHDx7E+fPn8cUX
X+Dbb7/FqVOnMHz4cJlBbUSEw4cPQ0tLCwKBQGnQ4PtUX1+P27dvQyAQ4OXL
l6zw7ohImlqcYRh07NgR7dq1U1hHR0cHCxYsgK6uLjZt2oTg4GAYGxsjIiIC
06ZNg62tbas6X375JWbPno0NGzbg4sWLGDRoEAwMDHD69Gm57fB4PDQ0NKCx
sRFff/01Fi9ejObmZgQGBkJbW1tmnR9++AFZWVnYt28fVq1ahZ49e2LChAm4
desWq4C7ESNGoKWlRYpaLou0tbURGRkJW9v/x957x1V5Zevj6z2Fcw69F6kB
IkSJMIrCABEZwDI2GLEQFOUiKqOCjrFFwcLYL6ASu2KQqNgJBBUlqKBioagM
KlWEQUW6KAKCz+8Pf+d8VU55McncmXuzPp/1D+x99vvud++91l7lWZbU2NhI
nZ2dkoCP6OhoEolEdObMGalRThwOh169ekW1tbXE5XJJKBTS6dOnqaioSG4J
BTFpaWmRpqamQpgnhmFIXV2dVFVVicPhEI/HI0NDQ5owYYJCVOY+ffrQkiVL
aOfOneTp6Unnz5+nNWvWyEzCZBiG+vTpQ15eXjR58mTy8fEhHx8fmjVrFunq
6kqd9zlz5pC7uzt1dXWRo6MjzZ8/nxYvXkzfffcd6erq0h//+Eeytrbu0U8M
ISVONNfT0yN3d3cqLS2VCxnT2tpK4eHhdOjQITIyMqKHDx9STU2NzPYCgYC8
vLzo6tWrdP36dbKysiJjY2O6ceMGubu70/HjxxWeM0TvAnWsra0pMzNTYdv2
9naqr6+n+fPn0w8//EALFy6UG7EpXkfd3d3k7u5O3333HdXX1yuEsgJAZ86c
oZSUFNq1axcJBAKpuITvE8MwNHbsWBoyZAi5uLiwDyD4/6nXcaTNzc10+fJl
+uGHH+jRo0f01VdfUXp6utxNzDAMBQQE0FdffUWLFy9mjRjQ3d1NQqGQ2tvb
qbGxUWF7AwMDsrGxocjISEpOTqYdO3awrvFA9A6+XFlZmTQ0NORma4thPmJj
YyUlC/Ly8uT+9okTJ6itrY3+8pe/kEgkIisrKyosLKSDBw/SiRMnZM5fSUkJ
nT59mrZu3Upr166l/Px8Vu/CMAyZmZnRqVOnqKWlhb799ltatGiR3Ixhca0W
AMQwDLW3t8uEYCd6F2ETHh5OAwYMoMuXLxOfz6eOjg46deoU5eTkUGFhYQ/U
Cj6fT4sWLaKamhqKj4+XwISkpKTIRQEAQCYmJqSurk6TJ0+m9evX0z/+8Q8K
CQmR+f6Wlpa0YsUKqq+vJzMzMzp8+DA9evSIxowZo2D23kWhDRw4kKqqquTC
tQsEAjI0NKTu7m76wx/+QIcPHyYAJBQK6bPPPqPMzEyZykhZWRm9fPmS5s2b
R9u2bSMOh0N2dnY0btw4OnbsmFQI+/dJRUWF3rx5ozDj/c2bN/TgwQM6ePAg
tba2Eo/Ho+7ubrp8+TJdv35dJn4YwzA0bdo0Kisro2fPntGkSZOotLSU7t27
J3O9Kikp0erVq2natGnU2dlJL1++JAMDAyJ6h2gwc+ZMSkpKotu3b0uE6v37
9yk7O5ssLCxo5syZxOFwSFtbm9ra2ig3N5fMzc2pb9++PTAR6+rqKDk5mdzc
3OjZs2cUFhZGgwcPptjYWLkCu7u7m27dukW3bt2iCRMmUFFRkUIUBABkZ2dH
kyZNokGDBpGenh4tXbqUXFxcqKysTGZ/MZ6fUCikkJAQamlpkQvaKaaioiKa
OHEiaWpqkpubG0VFRVFRUZFCEFRdXV3asGEDaWpqsq7Z9fbtWzpz5gxNmDCB
NXbikydPyNramtasWUN8Pp8iIiLkAh73+AG25jWhUIioqCiJjXDQoEESe/TH
bd/nIUOGID8/H35+fr2OEnNxccHBgwdZm0Pu3bsHDw8PJCYmIi4urlcRWN7e
3sjJyVFoD/by8kJhYSEKCwsxcuTIXtdyX7hwIdra2mRGun3MXC4XM2fORHJy
slwY/4/5wIEDiImJgbGxMY4fP84qSMLd3R0NDQ3o7u5GU1MTJkyYINOOP3bs
WFRWVmLSpElQVlbGzp07kZCQIDciT1tbGzU1NVizZg1EIhGWLVuG0tJShZVa
eTwedu7cifDwcERFRUFJSQmamprYsmWL1OdjGAaHDh1CYGAgNm3ahEuXLmHr
1q1ITU1FYmIinJycoKysLHNdqaurIzs7+wPTjTQWlzIeN24cZs2ahdGjRyM+
Ph5tbW1obm7GsGHD5JbV8Pf3R3V1NX7++WfU1taiqakJjY2NGDNmjMJxxc5s
NuuPy+XC1NQUDg4OsLOzg46ODpSVlbFw4UKZe0QkEuHWrVuYPXs2jh8/jqys
LOzbt09u1KCvry8aGhokpdUvXryIV69e4ejRo1izZg2OHz+Obdu29TD1qaqq
wsTEBBYWFpg6dSrq6uoQFRUFDQ0NueWh+/bti9jYWERHR8Pf3x+ZmZlYsWIF
qzkxMDBATk4O3N3dFbZ1dnbGyZMn8ejRI7S3t6OjowM1NTWIj4+XaY5SVlbG
pk2b8PDhQ1RVVaGxsRGrVq1iVTpA/M2srKywdu1aJCYmstq/ISEh6OjowPHj
xxWWGXmfNTU1kZqaiuzsbFbRdYaGhrhx4wZ8fHwQExOD77//vsd4suRKr246
/fv3p88//5xqamqIx+PR4MGDqa6uTqZ0ZBiGXFxcaNWqVXTu3DlKSUlhXdZA
TEZGRtTR0cEaf4jH41F8fDz94x//oKVLl/YKC0xcGVIeftOf/vQnio2Npfz8
fNq0aZME9ZktdXd3k729PT158oTOnz8vt6341tXS0kLJycnk7+9Po0aNokOH
DrEa6+rVq/Ttt99Se3s78Xg8VqCVDQ0N9OLFC9LU1CQ1NTUKCQmhq1evSjWJ
enp6UmpqKqWmplJnZyeVlpbSkydP5N6m9PT0SCQSkYWFBX333Xc0ceJEWrly
JZWXl8t9rq6uLoqPj6cdO3aQqqoqCYVC0tXVpUuXLknVMgFQRkYGjRgxgh4+
fEghISFUWVlJOjo6tHz5coqPj6eXL1/SwYMHpWLUdXR0UG1trUJTA4AeoLTP
nj2jsWPHUkZGBt26dUsuRtnx48epubmZli9fTkREeXl5kkJcisY1MTGRi2f2
PokL+71vihOJRHLX+tu3b6m4uJjCwsKovr6ezp49S/Hx8RK0eWlka2tL6urq
9Pnnn9Pnn39O//jHP+ivf/0rHT9+nF6/fk0CgYAAfHC7AiCpLiwSiejrr7+m
3Nxc2rlzp0wEcTGVlJTQN998I3mfwMBA+uc//8lqTw4ePJhEIhErnL0bN27Q
tGnTyMjIiH788Ue6evUqbdq0iZ48eSLzpvj555/T9OnTSVdXl9ra2ujOnTvk
5+dHQ4YMoTlz5vQwi2ppaZGSkhK5u7uThYUFGRkZkampKeXm5tKCBQsUWnus
ra1p8eLF9PDhw16Dpvbv358qKyvpxYsXZGdnpxBo+fnz53Ty5En661//SllZ
WWRvb0/a2tqsxuyV0GltbSULCwtas2YNMQxD/fv3l8DFSyM9PT0KDAykAwcO
0JkzZz6pAmV3dzcreynRO/NYeHg4WVpaUkpKSq/qYxCRBNJCloDT1tamwMBA
io+Pp4MHD/bKxyImLpdL6urq1NXVpVAgGhoa0po1a8jQ0JC4XC7179+ftb+J
iOjUqVM0fPhwmj9/Pu3evZsVGF9xcTHt2bOH5s6dS3w+n5qbm2WaQ7/99tsP
yizExcXImFKmAAAgAElEQVQpTHyrrq6mU6dO0ejRo+n169e0dOlSio+PZ/U+
eXl5NH78eBo8eDAZGhrS2bNnJSCq0uiHH36gY8eOUWdnp6RNbW0tLVmyhExM
TMjU1FSmEtTR0UFZWVn0xz/+kY4dO9Yr5UUkElFbWxsdPXpUIUhrd3c3nTt3
jq5evUrKysr06tUr1ujPOTk5kkqln0Jv376VW1Ono6ODZs+eTaqqqvT69Wt6
+fKlwrHu3LlDP/30Ez18+JAuXbpEt2/f/qCirqJ59PLyoi+//JKCgoJY71/x
NzQxMaF+/frJFYrv0+DBg+nOnTus279+/Zrq6uqotbWVsrKyFJo/a2pqKDEx
kfr160eHDh2ic+fOkb6+Pg0ePLgHoK6SkhIdOnSIurq6qKSkhF6+fElZWVl0
9epVamhoUKh0c7lcCgoKIiMjIwoMDKSioiJW7yQmoVBI9vb21NbWJreMu5je
vn1LO3bsoPLycrK1taVvv/1Wrl/sA+qNeY3D4cDR0RHh4eHw9fVVGBGlpaUF
R0fHXkWDfcxjxozBjBkzPrl/b9jCwgK5ubmws7OT+n8OhwMVFZVehVR/zAzD
YOnSpUhISGAFFqirq4uxY8di+PDh8Pf375V5jeidOcvHx6dXkW88Hg+amprQ
0tJilVTaWxaJRNDS0oKGhkavQQn/laykpPRJIcJ8Ph/29va9ziPqLXO53F80
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
gYG4f/8+srKycPfuXRQXF6O4uBiPHz9GS0sLqqqqsGrVKqlz6efnh7KyMrS3
t6OzsxPt7e2orq5WiGconneRSASGYeDj44Pm5mZER0fL7WNhYYHdu3ejuroa
cXFxMDMzY31D19XVxdGjR9He3o7g4GCZ7WxsbHDq1KkeZ9bx48fh4+OjcBwx
uC2b2zaXy+0hyJycnHDu3Dm5t1OGYSSlp21tbeHj44PCwkJERUX1mA81NTXs
2LFDUsJEX18fCxcuREFBAXx8fBTOn4aGBhISErBp0yY4Ojri5MmTmDJlisz2
4hImW7ZsAZfLRXh4OPLy8qQK7N8kek1FRQUXLlzAzJkzpU7cqFGjcP/+fVy+
fFlyqCgpKWHHjh0oLi6WetAwDIPQ0FDk5uZKpGm/fv3w9OlThZDlCxcuRHZ2
NmbOnImCggKsXLlSrsZpYGCAs2fP4sSJE1BTU8PChQtx5coVKCkpgWEY1gve
zc0NOTk5CsOZ+Xw+EhISEB8fj0GDBim8+urp6cHX1xfOzs4wNTWFqakpzp8/
j1evXmHUqFFy+44aNQpHjx6FiYkJPD09WWuZSUlJrAD/DA0NkZGRgW3btsHC
wgLq6urQ1NSEtbU1rl27JhepQlVVFcnJyVi0aBGioqKQn5+P69evswpfdXR0
RFNTE44ePcpK6CgpKSEyMhJ2dnaIiYlBYmKi3Hnn8/kICwvDgQMH0K9fP0yb
Ng1paWnIyclRWIaC6J1SMWfOHFah7dOmTcPp06chFAoxbNgwlJSUyDU1vm+G
5XA40NHRgZeXF06cOIH58+dLNT9xuVwYGBhAVVUVurq6MDQ0lNTU2bBhAzo6
OhAdHS11n3C5XJibm0v2VUdHBxobG+Hh4QEejwcej6dwjxgbGyMnJwddXV1Y
tmyZzHb29va4efMmurq6sGfPHowaNYpVBj6Px8OwYcOQkZGBjo4OVFdXywSp
5XA4WL58eY/yDA4ODsjOzma17p2dnVFYWPgBsCtb5nA4Cktr8Hg8fPPNN7hx
4waKi4tRUFCAlJQULFiwQKZSKxAIoKKighkzZuDGjRtoa2tDTk4O5s+fDw8P
D7k3F19fX5w6dQrq6urQ0dFBWlqaQjQMT09PFBUVwczMDLNnz0ZJSYlU0+kv
RiTgcDgkFApJJBJJgCFVVVVJU1OTDAwMiGGYD5x8NjY2tGvXLtLR0aF9+/YR
j8cjW1tbcnV1JX9/f5k4PTwej5ycnKiqqkqCWWVjY0NcLlchsGFlZSWVlpZS
cHAwWVlZUUFBgdykKnd3d3JwcKBx48ZRZ2cnDRkyhDIzM4nL5dKcOXMoPz+f
rl69qnBuXFxc6Nq1a3JBBolIkon/6NEjcnNzIy0tLbp27ZpMpIC6ujo6c+YM
Eb3D+Prb3/5GQ4YMoZ9++omuXbsmd6wLFy6Qjo4OxcTE0MuXL2nnzp0K38PQ
0FCSFc3lcuU6R1+/fk2rV6+m3Nxcam9vJ6J3WGB/+9vfqLm5mXbv3i2zv7Oz
M+no6NDbt2/J3t6epk2bRn379qU5c+bQ7du35SZHVlVV0fPnz1lXRFVSUiJX
V1caMGAAdXZ20vXr1+W+15/+9CcKDQ2lPXv20LJlyyRICPPnzyc1NbUeSX0f
U01NDT169IimT59O3333nWRupNGlS5coLCyMwsLCSE9PjwwNDWnixIkUExMj
1WHe1tZGDMOQmpoaDRw4kObNm0fm5uakoqIiAbd9/PjxB327u7sl2HHvz6ue
nh55eXlRe3s7Xb16Veo+efv2LXl7e0vAOhmGIVVVVdq4cSOVlJQQADpx4gSl
pqZ+0E8oFBKfzycDAwPatGkTDRo0iGpqauSiLPj4+EiCDr7++muaOHEiFRUV
0cKFC+UGIPz5z3+mHTt20LNnz+jVq1eUlZUlF6WBx+ORm5sbVVZWUkdHB1lZ
WUnw3hQlN/L5fAoLC6OnT5+Si4sLKSsr0/Xr11mhpRAROTg40BdffEGxsbEy
AyK4XC5pa2sTABIIBNTW1kbPnj2ju3fvSr7/x307Ojros88+o8WLF1NeXh6d
PXuWWltbSVtbm+bOnUsdHR20ePHiHhh9DMOQh4cH1dXV0fjx4+nrr78mKysr
hfiOeXl5VF9fT6mpqfT27VsyNTWlr7/+mqKjo9nNBdubjrOzM27cuIHKykqU
lZWhqqoK1dXVEk3JwMDgA23J1dUVtbW1aG1tRXNzM54+fYrnz5+jtbUVra2t
aGpqwqFDh3pIR7HpRVyrx8PDA3fv3sW9e/dYaY98Ph/+/v64f/++wiSpefPm
obq6Gi4uLrC2tkZFRQV27tyJjRs34vnz5x9UgPyYORwOlJSUYGFhgZs3byrU
DsTvZmtrC29vb6xduxZ5eXlYunSpQvu/2PzS0tLCukok0TtNdfHixcjLy2OV
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
wF6TXTZQCvH5fLKysiIvLy/y8fEhGxubT34gDodDXC73k/v/VtS3b1/y9vYm
gUDAqr2amhoNGzaMhg4dSkpKSjLbMQxDDMPIrdT4f4nEc8HhcFjn3Pxfod7O
CY/HI2dnZ9LX12fV3tLSkrZv304aGhq9fi4ul/u/5nvZ2dlRWloaWVtb/0vG
E8/fp54B/25nh3gP93o99Ma85uzsjKqqKrx+/Rrd3d24c+dOr66lDMPAwMAA
U6dOxffff4/8/HxWvgAVFRW4urqyut7z+fxPNgPo6+sjOzsblZWVcqFwxD6u
kJAQJCYmYseOHThx4gS8vLyktrexscGGDRuwa9cuxMfHY+nSpXByclJo7uHx
ePDw8ED//v3h6OiIkSNHwtvb+zevxaOtrQ0HB4ffzCxnYmKCLVu24OTJk/j5
55+xZMkSWFpasvpuysrKMDIy+sU1YMR+K3ltuFwuTExMYGNjAzs7O4URQJqa
mhg+fHiv/Gcfs62tLfbu3csqfJzo/+WlVVRUyI16E7MY0HX//v29Aru0srJC
YmIibt++jaVLl8LOzk5heDKHw/lF+/G3ZLGf78WLF70Gj/0UVldXx/fff4+C
ggJWfkINDY0PCqqJ88wUuRnE7ogpU6agf//+cr8xl8uFs7Nzr0KwxXNnZGSE
qKgopKeny/QP/io+HXNzc9y7dw/V1dU4cuQISkpKkJiYyGrxmpiYYNmyZSgq
KkJNTQ2uXbuGuLg4hSB7HA4Hs2fPRnp6utzDlsvlYvjw4diwYYPE/yGOw2c7
kYsXL0ZLSwtiYmJkvpM4CfLIkSNYsWIFrK2twePxMG/ePEydOlVqnyFDhiA8
PBxubm7w8/PDqlWr8PPPP2PPnj0yk8u0tbWxaNEiVFRU4MyZM5g7dy7c3NxQ
Xl7OCoKDYRiYm5tj6tSp8Pb2VphIOWrUKImj3cPDA9nZ2azChMVJsgKBACKR
CCoqKlBWVpZ50KiqqiIpKQnFxcVISEhARkYGGhsbUVFRAT8/P6n9tLW1MXz4
cCxatAjXrl3D48ePsX79etagpOJ8CXGIsEAgQGBgIDZu3ChV8AuFQjg5OWHX
rl24ffs2cnJyUF1djVmzZskdx9PTEzU1NTh8+PAnoRdbWlri2rVryM7OZhXO
zOfzMX/+fDx69AhhYWGsfFYeHh4oKSnpla9KV1cXmZmZuHbtGmbMmIGEhATk
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
w8GaNWuQnJwMKysrVhhlU6ZMQWNjI6qrqxEWFsYqik9cGyc9PR0rVqxQWJLd
1tZWYm4Woz2npqbKNS3KGnfz5s2sSmt8zNra2khOTpY67+rq6pg/fz4SExOR
np6Ow4cPIz4+HgEBAQrnkMvlwtfXFz/++CMuXryIsLAwVuH32tra0NfX75XP
18HBAUVFRaivr0dnZyd+/vlnVtGnQqEQaWlpuH//Po4cOYKhQ4eyuulMmTIF
oaGhUFFRwdatW7FmzZpeP++LFy/w+PFjmaZ1Gxsb5Ofno7W1Fd3d3SgvL8fQ
oUN7tPtVYXB0dHTIysqKGhoa6PHjxzIlnJKSEvXt25c8PT3JzMyMOjo6SCQS
0eHDhyk/P18h2vTIkSNp9erVNHXqVCorK5Pb9mOaNGkShYaG0vjx4+nFixdy
24rf5+XLl1RSUiJXYjMMQzY2NqSqqkolJSWS32YYhjQ0NKitrY06Ozt79NPU
1CRbW1syNjamzz//nIRCIb19+5by8/MpOzubWlpaevTR19cnHx8fcnd3JxMT
E2ppaaHKykrKycmhc+fOUXNzs9z3sra2psOHD9OECRPkVlN8nwwMDIjH49Gz
Z88oJiaGdHV1KSgoSOo7fUweHh4UGxtLTk5OctGwVVVVycHBgSwsLGj06NGk
rKxMFRUVlJ6eTpmZmTLHMjMzoxUrVpCysjLl5+fTxYsX6cGDB3JvsyoqKpSQ
kEBv376loKAgevXqldx3sLOzo9DQUDIzMyMiooKCAjp9+jTdv3+f1RwQvUM6
P3XqFC1ZsoQuX76ssP3QoUPp888/p/j4eFY3JD6fT1988QW1t7dTaWnpbwLj
YmZmRn//+9+Jz+fTixcv6MmTJ5SYmEgVFRW9+h19fX1KTU2luLg4SkpKYq8N
E5G9vT1FRkbStGnTqK2tTWobHo9Henp6BIDevn1LjY2NvRrjtyaGYcje3p68
vLyoubmZLl26ROXl5Qr7qaqqUlRUFDU2NtL+/fvp6dOnrMZzcHCgqKgoev36
NVVXV9PmzZvl3pw/JhUVFYqMjCQHBweaMWOG1HGVlZXJ19eXRo8eTQ8ePKCj
R49SeXl5j3UoCwbnk0Km/1UsFAp7VCRly9ra2pg7dy7rBMJ/Z+bxeDAwMICB
gUGvMvGHDRumsBLlxyxO8CR65yg0NzdnnbQXHh6OjIwM1gm5/wpWV1fH+vXr
5YaPfsx8Ph8aGhqfjPzAMAxMTU1/c+SI/wQWCARYtGiRzGAPeezp6SnX//E7
92SGYWBpaQlzc/NeI+2LWSQSQVtb+xfP+6960/md/jNIJBKRiooK1dfX/0vG
Gzt2LAmFQjp58uRvon3/Tv+3SOzjU3Sj/53+PUnWTed/ndDh8Xg0ZcoUKiws
pLt37/5PP87v9Dv9WxGHwyGRSERtbW2/Kwa/029KsoTOvxds6a9ABgYGEj/Q
vwIN19HRkSIiIigsLIwVaq+WlpYkIo3L5ZKfnx8rlFsej0c6Ojqkrq6uEJ1b
IBCQqqoquxf4iGxsbGjKlCnk4+PzL5k/LS0tcnBw+KS+QqGQxo4dKxfd+9ek
QYMGkYWFBau26urqNHPmTPL29mb9fHw+n6ZMmcJ6DCIiX19fGjBgAOv2dnZ2
dP78efriiy9Y9xGJROTk5ERhYWE0ffr0Xs03l8slTU1N0tXV/bdEle8t8Xg8
0tLSIpFI9JuOIxAIaOLEieTq6kp8Pv8X/Za5uTn16dPnV3qyd8TlcklDQ4P0
9PRIR0end/PB1qfD5XJhamoKe3t7+Pn5YcyYMQp9LSoqKr9acS8x0oCOjo7c
jOjQ0FC8evUKQUFBvbaFMgwDkUjE2hZva2uL69ev482bN9izZ49MnCMx6+vr
4/jx4+jfvz+I3kWIXb58WS4gIo/Hg4ODA2JiYnDjxg1cu3YNU6ZMkWlvVVVV
RUxMDJKSkno196qqqggODkZpaSlev36Nc+fOKXyfj+dPRUUFffr06ZUfbeTI
kUhNTWXlB7K2tv7g3b28vHDw4EHW4JoWFhaYNm0aQkNDYW5u3qv1YWRkhIyM
DFb5VJqamti5cydev36NpqYmufkp77OpqSlKS0t7Fa25adMm7Ny5k7X9PSAg
AOXl5ayRsDkcDpYtW4bHjx+jqqoKtbW1rAFnra2tERMTgzt37qC8vBx+fn4K
kxvt7OwQHh6OFStW9BrpXFVVFVZWVnB2doaBgYHcfSwUChEUFIQtW7Zgy5Yt
WLlyJVxdXaGvry81r4VhGJiZmWHDhg148OABYmNjWfsubW1tsWzZMtjZ2bE+
W3R1dREZGYkLFy70GrX8/XNZSUkJ27Ztk4k6wefzoa2tDR0dHejo6CiEAeNy
ufD398e2bdtw+fJllJaWorCwEKdPn0afPn0+aCtLrvRMApFBw4YNo3nz5tG9
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
jY0y9bsMw0AgEMDExAQRERHo6OjAzp07JW4sDQ0NnDlzBvn5+WJ7ieiVuWTJ
ElavAl9fX2zbto3VN3A4HJiZmcHHxwcXL15EZWUldu3a9cHl+j6Hh4ejvb0d
cXFxOHXqFHJzc9HW1oaXL18iODhYpmChqamJYcOGYebMmSguLsa4ceOkriE1
NTV4eXkhOTkZNTU16OnpQW9vL9LS0jBmzBjo6urKlLwnTZqEgIAAbNy4EQkJ
CayjyNXU1JCUlIRbt26xPmgtLS3F2XvHjx+P4uJiVq+kIUOGwMvLC8+fP0dD
Q4PMOjY2NiguLv4g4Zi6ujpycnJY5bYiemejefz4MWbPni21jKKiIvbu3Ysr
V65g/PjxOHDgAKvxEyGPi+aTy+XCx8cHWVlZrOwn3t7eqKmpgbe3NwoKCljZ
Z4jepRsR2VrZlP95XWmXjra2Ni5fvoydO3fC2toaXl5eiIyMxN27d+Xi673P
VlZWrHIrSRvTkSNHIj09Hbt37xafF7/ae+3nNGrUKFJSUpLqFSGin+sH+/v7
qaenhzQ0ND4qyzAMubi40Pbt26m7u5ueP39OHR0dpK+vT0KhUCr+EhFRZ2cn
nT59miwsLGjNmjVUW1tL3t7eFBcXJ1Vf29vbSzt27CBLS0sqKSmRi+Ml6n94
eDi5uLjQlStXaPjw4aSvr08nTpyQWU9Ejo6OZG9vT1lZWaStrU21tbVydbx8
Pp9++ukn6u3tJXNzc/rjH/9IkZGRrNoTEQBSV1enkSNHUnZ2tsQ2vLy8aO7c
ufSHP/yBDAwMKDExkSIiIiT27/Xr1+Tt7U0aGhokEAjIwcGBANC9e/dY47wZ
GhpSZ2cncblcueM+MDBAFRUVdPDgQZo8eTIpKSnR4cOHpa6JY8eOkaGhIVlb
W1N3dzfl5ORQeHg4ZWRkUEVFhUx8rpaWFuJwOOTh4UGxsbGUnZ0tcQ0pKyvT
8ePHxfbDhoYG2r9/P1VXV5O7uztFRkZSamoqHTp0iGpqaiS2lZmZSXfv3qWd
O3dSbW0ta3vOH/7wBxozZgxt2rSJNdbY48eP6fTp07Rp0yby8PCg69evU3x8
vNTyxsbGNHXqVPLw8CBra2t6+vQprVmzRiZadENDAzU3N9PkyZPpypUrxOPx
aP78+aSiokItLS2s+snlconL5VJtba3UMpaWluTp6UlpaWm0e/du+uSTT+jY
sWMyPSCJ3q3zrVu3UmFhISUkJFBfXx8lJibSxIkTWfWts7OTamtrKTY2lgwM
DOgvf/kLPXjwQK6N+quvvqKYmBjWe4NhGNLU1CQNDQ3y8PCga9euSSzX1NRE
7u7uYk/bJ0+e0Jw5cygnJ4d1WxwOhxYsWEAVFRV07949VnVEpKCgQBoaGjRy
5EgaGBig169fy7WP/aJLh2EY+uKLL+jp06cy3YolkaKiIhkbG0us19vbS4GB
gaSvr0/m5uY0dOhQampqosTERFZQLlevXqXMzExSVFSkzs5OamhokDsAJSUl
VFJSMqhvqKyspG3bttGXX35JnZ2d9P3331NMTAyruvb29tTT00PZ2dl07tw5
uYZSoncXN4fDITc3N5o5cybdvn17UG6Uvb29lJycTObm5lIPKIZhSCAQkLGx
MQGgoKAgOn78uEx31a6uLvH/V1dXs+6PiDIyMmjevHkkEAgGtY6UlZXp+fPn
MusUFxfT/PnzSVlZmfr7++nNmzesATuJ3gEu6unpUWBgoNQDBQA1NjbS1atX
KTU1ldLS0qimpoYGBgbo2rVrpK2tTa9evZJpmAVAfX191NTURJqamqz7p6ys
TBUVFazSfohoYGCAjh49Ss+ePaOamhr6+9//LtPd/NNPP6Xdu3dTZ2cnfffd
dxQRESHXRbiuro78/f3p0KFDtHbtWhIKhcThcMjPz4/y8/NZ95XL5ZKHh4c4
9cPP6enTpxQSEkLGxsaUkpJC6enplJ+fL1eA6+3tpatXr9KRI0do6tSplJ2d
TX/84x9JSUmJlTPGrVu3yMnJiU6cOEFaWlp0584dufWmTJlCysrKdOrUKdZC
BYfDIVdXV1q+fDldvnyZfvjhB6ll398H6urq1NnZSRkZGawRJxQUFMjW1pby
8/PlOkfw+XxydHQkDQ0NcciDpqYmPX36lHbt2kUPHjyQ3+4vUa8xDINt27Yh
KipKrjrq56ylpYWYmBhW8N7/7iwUCgcdAKuiogJtbe1BAXeK3Dy9vb1hb2//
iwJuVVRUMGPGDJnzJUqEp62t/YtAVgfLfD6fdbqA91lRUfEX1RsMMwwDNTU1
uTpxPp//mwCcKioqDsqeIwrq+2eCYfJ4POjq6kJLS2tQ65XD4WDkyJFYvXo1
lixZAjMzs0GtJ6FQiKioKLi5ucn8PpHr7i+ZW5FrsShl87Rp01iPpYaGBlat
WoVNmzbJTdSnqqqK2NhYuLu7D3quRHM82LNCWVl5UONibGyMZ8+esQqoFwgE
WLlyJbZv3465c+fCysoKGhoaEs+k3xzwUyAQEI/Hk6nykkRKSko0ceJEVhLC
7/Q7/U7/mSQUCqmnp+efmqZAUVGRhEIh9ff3U1tb2z/lPGIYhtTU1Kijo+Pf
KuXC+yQQCGj27NmUl5c36NQVsug/BvDzd/qdfqff6Xf636f/GMDPf3fS1NQk
AwODXw1+yOVySVVV9Tfq1a8nhmFowoQJ5OXl9U8HQ/x/gfh8PhkbG5Opqem/
HdCloqIiGRgY0IgRI/6t1tD/FpmampK/vz95enrKBbm1s7OjiIgIsrCw+MXt
MQxDGhoaZGZmJhc8dcSIEVLTRr9PPB6PzMzMaPTo0aSmpvaL+8aGeDweGRgY
kIGBwS8CBR60IwHDMPTpp5/StGnTaOzYsRQXF0fR0dEyDXhcLpeMjIxIXV2d
6uvrZXqlSCIOhyPOzS5PnaehoUGffPIJcTgcGhgYoNraWmpoaJDZPxMTEzHe
mL+/PyujtmiSm5ubqbGxkfW3rF69mhwdHenEiRNSPVLY0GeffUaLFi0iX1/f
X/wb79OQIUPos88+I1tbWxo7dixxOBwqLCykK1eusMICU1RUJA8PDzIyMqJn
z55RcnIyq3b5fD598skn4o0iwqITvcBFG5TL5VJbW5vMjKS/NTEMQzo6OqSr
q0s1NTWsnD6I3qmGfH19acqUKcTlcmn58uVUVlYmtfzo0aPJ2dmZrl27Rt3d
3WK08paWFmpoaJDaN11dXVJWVqb29nbWa9DU1JT27t1Ln376KSkoKFB5eTmt
WrWKXr9+LbWOvr4+rVy5kkxMTOjNmzfU1dVFISEhcr2jlJWVyczMjPh8PvX2
9tKzZ89kOlX8UlJSUiIHBweaMWMGNTY2UkpKCpWVlVFHR4dcldno0aMpIiKC
MjMzKS0tTWZ5DQ0N2r17Nzk7O9PNmzeptbWV6uvrWanlhEIhmZiY0IgRI+hP
f/oTOTk5UUtLCy1ZskSmg4uPjw9paWnR4sWLZf7+n/70J7py5QqpqqrS1atX
acOGDVIRI34tzZo1i/bv308Mw5CPjw9dv359cD8wGJTp4cOHY/369Xj8+DEy
MzNx8uRJFBYWyox5MDAwwO7du3Hq1CmEh4fj4cOHsLKyYm2QnDBhAiIjI5GS
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
prIW5CwtLVFfX89aoCUiLFy4EO3t7awvnSVLliAmJgY2NjZQVFSEn58fsrKy
sGfPHpnnII/HQ1RUFLKysqSe57/60uFyuTh16hQGBgbQ3t6Ox48fS8VfEvHO
nTvx8uVL5ObmIiIiQgxaySbORAQ/XlJSgujoaEyYMEGmxKiqqor9+/ejrKwM
KSkpmDlzJitIkUmTJiErKwuHDh3CzZs34e3tLbcOj8eDpaUl+Hy+GJJ+8eLF
rBaFsbExjhw5glu3bmHs2LEyy4riCZKSkjB69GgwDIM1a9Zg3759rGN1DA0N
8fTp0w+k25+ztbU1goKCkJWVJYb/FyWdGsyLbMaMGTh9+rTUvg0bNgwZGRlI
T0//6IVhbW2N6upqmXDxampqOH78OJKTk5Gfn48VK1bITBhnaGiI3bt3o6ys
DK2trQgPDxe/okRAjJKkOV9fXzx//hxubm6YNm0aTp48iba2NqSnp8tUsXE4
HFy4cAEeHh6sxo1hGPj4+KC1tRWXL19Gfn4+kpKSpEKl8Pl8hIWFoba2VqJg
ZWNjg4aGBuzcuVNifSUlJSQnJ4sPv7KyMgQEBEiV9OfOnYvq6mocP34ct2/f
xpMnT/DmzRsMDAzg0KFDUiVhJycnFBUVYd26dbh06RIaGxsRHR0tU91lZmaG
4uJilJSU4N69e+jo6JB5xohAiPv6+tDe3o65c+eKU3lcvHgRfX19WLlypdT9
GxAQgOzsbCxZsgTr16/H5MmTpZ4vBgYGKCoqQnV19QfCnlAoxKNHj5CcnCxV
PcflcjFjxgyMGTMGlpaWCA8PR2RkJCttipOTE16/fs0qcaBoTEJDQ9Hf34/q
6mqpGigRq6io4MaNGxg/fjz09fWxceNG5ObmYty4cXIfBgzD4OTJk7h9+7bU
ef3Vl86YMWOQn5+P2NhYLF++HDdu3IC/v79MvCJtbW2MGzcOhoaGEAqF2LJl
C6tLR0dHB8HBwWhvb0dBQYFUIE1Jm9LQ0BBBQUGorq7Gjh07WGExWVhYIDMz
E+3t7Zg/f/6gAqumTZuGvLw86Orqsq7D5/Mxe/ZsJCcnS8UoU1RUxJw5c1Ba
Wop9+/ZBKBTCxMQEV69eZT0eRAQ3NzfU1tZKtaNxOBx4eHhg8+bNMDIygp2d
HVxdXREUFITc3Fxs2bJFpr6by+XCxcUFrq6uCA4OlonLtWTJEhQWFmL48OEf
/F1ZWRlXr15FTEyM3MXO4/GgrKyMJUuW4NGjRzh06JDMVwWHw4GBgQEWLVqE
hw8fIjIyUm4bzs7OqK2tRXNzMxobG9Hf34/6+nqJOUPeZyMjIxQWFuL8+fMI
CAiAo6Oj3IDAYcOGYeLEiRAKhbCxsUF9fT38/Pwk1hMKhSgoKMCVK1c+OiC5
XC7Wr1+Ply9fSjykGIbB9OnT8ezZM/T39yMtLQ02NjYyL8dx48bh9evXaG1t
RV5eHm7cuIGMjAy0tbXh8ePHUnX5CgoK2LRpE5KSklBSUoLnz5/LRTtmGAbD
hg2Ds7Mztm7dKs7HI20vCoVC5OTkoK+vD42NjdDX1wfRuyydGRkZqKmpkaqF
GTZsGKqqqtDQ0IAbN24gPDxcrJqUVH7evHno6OjAiRMnPhgvUe6pFy9esMY5
MzQ0xI0bN+Rm1+VyuWIhmu3ZMnr0aFRUVKCvrw9nz56VG7SsoKCAkJAQpKam
IiEhATU1NZgzZw7rIFY7Ozs8ffoUCQkJEoWDX3XpKCoq4tq1a/Dz8xNvcB0d
HURERGD79u2sUw8EBQXhwoULMi8dQ0NDXL58GSdPnsTixYvx9OlTjBo1Su7v
vz9QCgoKWLNmDWpqauTqhInevT5E0P+PHj1iDXqnpKSEq1ev4ttvvx10tLGS
khJu374tFZlh3759KCwsxP79+3HkyBEEBgYiJiYGW7ZsYd0Wl8vFtWvXcO3a
NZmLnMPhfPSbfD4f48aNQ2JiIg4fPixV3ePk5ITz588jISEB9fX1mDp1qtSy
3t7eiIuL+6AvIrj0jIwM1jDsOjo6OHXqFKZMmYJLly5hy5YtrF5+M2bMQGNj
o1yQWi6XC2NjY9jY2GDv3r3o6urCoUOH5L5exo4di+fPn2PChAmwsLDAsWPH
pF7YAoEA1tbWH/zNzs4OTU1N8Pf3lzjHKioqqK6uxoEDBz74fxFacmVlJbZv
3y5RYrewsEBVVRUuXbqEuro6BAQEyB2vWbNmISsrC+PHj4empiaUlJSgo6OD
ffv2IS8vT6ZDEJfLhampKUpKSnD06NFBoWisWrUKHR0dGDdunNQyQqEQ+fn5
4ktn2LBhsLW1RUJCApqamrBx40aph66ZmRmampqwfv16qKiogMPhYPPmzdi7
d6/EvfH111+ju7tbnJLZ1NQUXl5eSElJQXt7O44dO/aRICMUCjFz5syPEPbt
7Ozg7+8v98xUUVFBcnIy7t+/z8ohQFlZGRcvXkRPTw/6+vpQUlKCtWvXwtHR
Uea+UlFRwdChQzF79mzExsayMn2ImGEYuLq6orm5GdeuXfvoASLtXmEVp8Ph
cKivr48aGxvFWE39/f3U399PQ4cOZeWrzeFwSEtLi8rKyqTiDykoKFBAQAB1
d3fTo0ePaMGCBdTa2srKVXXs2LHk4uJCHA6Huru76eLFi1ReXk48nnyvcIFA
QADom2++oaCgINq0aRMpKirKrefs7EwmJiZ0/vx51tHMHA6HeDwe2draUmtr
q1Tsq5MnT9Kf//xn2rlzJ23fvp2Kioro008/pe+//551WxYWFmRlZUWXLl2S
6QY+MDDw0W/29/fTjz/+SElJSfSnP/1JYuwNh8Oh2bNnU1paGnV2dlJ8fDwt
XbqUXFxcJLZTWFhINjY2tH79ejI2NiYXFxcKCgqiTz75hJYvX846EZSpqSlN
njyZhg4dSqmpqeTt7U2WlpYflbOxsaGhQ4eSkpISaWpq0qhRo0hBQYGMjIxk
/n5/fz9VVVWRQCAgNzc3un37Nh04cEAublZrayu1tbXR69ev6c2bNyQUCqXu
jU8++YT2799PdnZ2pK6uTqNGjaLDhw+TkpIS3b17V+IcDwwMUFNTE82ePZuW
L19OLi4utHLlSjp79iy5uLiQj48PHTlyRKILLofzbqvr6emRQCAgbW1tmd9C
9M7NNycnh3JycqitrY0UFBTI1NSUHBwcqKOjg169eiW1LgCaPn06DQwM0JEj
R1hH44vivX766SeZ0fEi/EKid273a9eupaioKJo2bRr98MMPFBoaKhX8s7W1
lUpLS8VxLbNnz6a//OUvEpMJDgwMUFFREbW3t9Of//xnKigooOzsbDp27BhN
nDj5e7H9AAAgAElEQVSRfvzxRzp27NhHoRxmZmYUHBxMf/zjH8nExIRGjhxJ
W7dupf/5n/+huLg4uRhxGhoaZGRkRD/++CMr1BeBQEAGBgbU29tLDMOQhYUF
BQYGUnx8PHl7e0uNFfvHP/5BLS0tNHv2bDpz5gxrhBmGYegPf/gDLVmyhNTU
1GjKlCly95WY2KrXnJycUFBQgFOnTmHfvn3IzMxEVlaWzHTB7zOfz0dkZCSi
oqKkGvoFAgGCg4NRVlaG+/fvIywsDFZWVqwkeysrK1y9ehXe3t6wtraGtbU1
cnNz5XpSEb2zFVy8eBGhoaFYs2YNHj16JBfmXFVVFfHx8di2bdugXjl2dnYI
CQlBUlISa1h0Ho+HkJAQXL9+fVCSyNKlS1FeXs7KoKimpgZ9fX0MHToUU6ZM
QUBAAKKjoxESEgJ7e3up3zh9+nQkJSVhy5YtUFRUhKKiolSplsfjYeHChcjL
y0N1dbXYpjAY1STRO4eChIQEvH79GvX19YiNjZX4jePGjcOZM2cQGxuL7Oxs
NDY2wt/fn5XkOHz4cGRlZeH+/fusX2BcLhf79u1Deno6rl69iu3bt0uVtpWV
lREcHIy6ujrk5+fj5cuX6OjowKlTp2SqC8eMGYOoqChUVlYiJycH58+fx8KF
C+XisHE4HEybNg3FxcXo7OyU6EAhaW7Lyspw/vx5nDt3TpwuJC8vT+7aHTVq
FJ48eYLVq1cPam7V1dVRWFiIq1evylUPiVx3+/r60N3dLU5lwUbV5eDggLNn
zyIzMxPnzp2Di4uL1HWroKCA5cuX4969e3jy5AmePHmCwsJCHDhwQGr4hshB
pL29Hc3NzaiurkZycjImT57MSjM0a9YstLe3s3YiEKWPXrp0KTIzM3Hnzh2x
V5+8FOhWVla4fv06q1xb76/f69evo6+vD69evUJISMhH9s5fbdPhcDiYOHEi
zp8/j5s3b2L37t2DykrH5XIRGBiI5ORkmV5RAoEAmpqaUFZWHpQRm+hdYrG7
d++irq4Oz549Q0JCAqvsfkTvfPYDAwORk5ODlStXylwYovietLS0QQOeKigo
YOrUqVLzwEhjU1PTQeXHEAgEiIyMxM2bN1kBUorcLYODg7Fr1y64urrCzMxM
pvOGaCykGeSlldfR0YGtra04GdRgxk/E2tramDZtGsaNGyf1EmEYBsbGxvD0
9ISXlxecnZ1ZXdpcLhc7duzAy5cvWblmv8+qqqpYsGCBTMO0iJWVlbFo0SKc
P38eSUlJ8Pb2ZrXxBQIBhg4diiFDhgwa/NXY2Bhz585lFXOkqamJwMBAFBQU
oKioCJGRkVi8eLFcr0ZRHFl6evqg98ewYcNQWVmJM2fOyC2rpqYGb29v5Ofn
o6CgAJs2bRpUezweDwoKCqzGUJRrSiRYyVvzDMPA3t4ey5cvx/z582FnZ8cq
X8/737Zw4cJBC2SidaWuro5x48bBzc0NDg4OMs8aDw8PqXmzpDGXy8XUqVPh
7u4Oa2triWfMr750RMzhcMQJ1gY7GCNGjGBln/k1rKOjg5kzZ2LevHlyvTck
fZsoVkXeArSzs2NlL/rfYg6Hg40bN2LPnj2D+vZfOrf/19jKygqOjo6/+FIc
7Fz9u467aF3weLxBCRZ2dnasXYPfZx0dHVy4cEGq55mktkT9+3ccv3935nA4
2LJlC2uty2BY2r3yO+Dn/3FiGIa1Deh3+p3+Hej3NfuvpX/WeOM/DfCTYRgy
NzenSZMmkYKCgsyyXC6XJkyYQGZmZv+i3g2Ofgmonoj+r21ePT09GjduHJmb
m7NyEvl/lRQVFcnCwuJXzf1vSaIkgkFBQYMady0tLXJyciITExPWdf6vrdl/
FamoqJCVlRUNGzZsUPX+5eM9WPUa0f+vDvhnqwSUlJRgYWEBJycnDBkyZFDP
+8WLFyM7OxvffPONXMPx5MmT0dzcjMDAwEHryDkcDoRCISwsLODp6YlLly7B
0dFRZh0HBwdcvnxZLjaUUCgUB7GFhIRIjen5LVhVVRU7duzAxYsXMX36dNZj
LRAIsGrVKoSHh8PPz29Qdqf31xKb9rS0tJCUlCQ2zF64cEGuCpXL5WL69Ok4
deoUjh49yjrGydjYGDdu3PjIrVkeKyoqwsTERKrbOBvm8/nYu3cv7t+/zxqn
8J/NlpaWqKyslBk783O2s7NDRkYGXr16JRePj8vlYuTIkVi2bBmCg4Oxdu1a
TJ8+nTWyg6qqKsLCwjBlypRBf5uLiwtmz57NuryBgQGOHDmClStXylS/MgwD
ExMT+Pn5iR2v9u3bx8qxR8Tq6uo4efKkXOxEXV1dHDp0CIcOHcLZs2cxY8YM
1vOkq6uLPXv2ICMjA+vXrx9UUkIulwt1dXWJtsjfxKbDMAwmTZqE6OhoJCcn
IzU1Ff7+/nKNs8OGDYOrqyscHBxYoQQQvQu0u379Ompra9HR0SGOWWHjLefs
7IwnT57A09NTrk5eSUkJMTExYt/2kJAQuLu7szLoimKVCgsLUVNTgydPnsDf
318uttKSJUvQ3NwsNfJcxGZmZqiurhaDl54+fXpQFw/DMBgxYgTmzZuHefPm
ybzkLC0t8ebNGzGkCBuvP6J3xlg3NzfMnz8fd+7cgY+PDytBRCAQwMXFBZcv
X8atW7dw9uxZueMhFArh7OwMS0tLeHh4oLGxUS6MiZGREVJTU3Hr1i28evUK
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
168lpgpWV1cnFRUV6u/vF+MntbW1UX5+Pr1580bmWBARTZ48mXR0dOjp06dy
yxIRjR8/ntzc3IjL5dLAwACNGDGCtLS0aMiQIRLdaQcGBujs2bOUnJxMRUVF
1NPTQ0eOHKG6ujpW7VVVVdHWrVvJ3t6ewsLCSE9PT2Z5JSUlqqqqorq6OgJA
JSUlpKmpSebm5lLrvH37lr766ivy9PQkVVVVioiIkNo/VVVVWrRo0QdroqOj
g3g8HnG5XJlplhmGIUdHR1JUVKSbN29SamoqWVlZScWh+jmVlpZSY2MjDR8+
XCpWGZfLpalTp9KjR4/o2rVrYvdTbW1t2rt3L61evZoOHz5M+fn5H9VVV1cn
oVBIBgYGpKGhQatWraJhw4bRzJkzSVlZ+aPyCgoKtGzZMtLW1iY+n09mZmZ0
48YNuTheIuru7qaOjg7xv4cPH04Mw8hct319feTs7EyLFy8mJSUl0tfXp76+
PtqxY4fENfz27VtKSUkhPT09EgqFpKCgQAoKCtTT00MAqKqqSmL6cz6fT3Pn
zqXo6GhSUVGh+Ph4UlZWJkNDQ7ku2jwej0aPHk09PT2s1zkR0eeff04Mw1B6
errcskOHDiUtLS1qamoS/+2zzz6jpqYmqfiJIhoxYgStWLGCuru7qbW1lb74
4guytbUlFRUVqXUYhqE5c+ZQRUUFVVdXy+0fh8MhQ0NDsre3JzMzM5owYQI9
efJEbr0xY8bQ0aNH6dKlSxQXF0dNTU20f/9+mjJlCs2cOfODsgKBgNauXUsK
Cgrk4+NDf/3rX6mgoIBcXV3p0qVLtGfPHvL39xdj4kmjQQU69PT00E8//US2
trYUERFBRUVFrOoNDAyQqakpqaioUE5OjkzgxNraWurv7ycOh0NDhw6lrVu3
0pYtW4iIaPHixZSXlyexHgB68+YN1dXVUU5ODnV3d5OFhQV1dXVJ9EN/8+YN
dXR0kKmpKZ05c4ZaWlrIxsaGhg0bRlwulwIDA+nvf//7B3UEAgF99913lJ+f
T6GhoUT0Lm7kyy+/JEVFRaqqqpI7FkOHDqWGhgaqra2VW5ZhGHJxcSFjY2Mi
eperPioqip49e0bV1dW0cuVKiQdGfX09ff/99+Tk5ESvXr2ijIwMuW2932Zr
ayt1d3fT+PHjaeHChRQcHCy17Lp168jQ0JD+9re/UUdHB82bN4+EQiGVl5fL
bKevr49sbW2ps7OTEhMTpZbT0tKiHTt20Lhx4yg+Pp5evXpFlpaW9PXXXxOX
y6W4uDipdTkcDn355Zc0YsQI+uGHH0hJSYny8/PlAneKqL6+nrKzs2nu3Llk
ampKL168kNg/BwcHun79uvjwGzZsGH399ddkampK/v7+FBERIbHNkydP0tmz
ZykhIYFaW1vJzMyMrly5QsePH6e2traPyvf29lJLSwvt27ePcnNzqa+vj16+
fMnqW35ODMOQmZkZZWdnf3CQvk8AKDMzk9atW0c8Ho+ampooKyuLgoKCKDs7
W+K+6unpoR9++IHmz59PFhYWpKCgQAcOHKBr167Rzp07KTQ09CNhjsfj0aZN
m2j79u0kFAqpvr6eRo0aRebm5qSoqEgvXrygkJAQamxslNjPvr4+qqyspD/9
6U8yQUjfJw6HQ+PHj6fKykq5gjMRUUtLC7W0tJBQKCSid6Cz27Zto4sXL35w
kf+cRo8eTVu3biVDQ0NiGIZOnTpFAKijo4MePnxI27Zto7Kyso/q6erq0uzZ
s2nHjh2swFINDAxo27Zt9OOPP5KFhQU5OTlRaWkpFRcXSwSAFdGECRMoLy+P
UlNTKSIigq5du0ZBQUEUFxf30ZrlcDikp6dHKioq9N1339E//vEP6ujooObm
ZkpKSiIfHx+ZgM5iGqzL9M6dO5GXlzcoX3PR8/fVq1dyvY2srKzw+vVr9PX1
ob6+Hr29vXj79i327t0rV5Vna2uL+/fvIysrC4mJiYiLi5MKVcPn87FhwwZU
V1ejt7cXvb296OnpwbNnz3Ds2DGJjg5KSkq4d+8eHj9+jMTERCQmJuLRo0e4
efMmTExM5I6Dtra22P7BNivn+PHj4e/vj8rKSjQ2Noqhy0tLS+Vi2B0/fhwL
Fy4c1LN81qxZePnyJfr7+9HU1CS3/qhRo1BRUYFz584hOjoaxcXFrHJyDBs2
DEVFRXKT5olUdgkJCSgsLERTUxPq6upQWVmJPXv2yPTAYhgGa9euRU1NDTIz
M1FcXDxoQ3NAQAD6+/ul6scVFRVx9OhR1NTUoKSkBK2traipqcGxY8dgbW0t
V/VlaGiIOXPmoKSkBJGRkXLzrDAMg+HDh2Pu3LmDjol6nzU0NPDgwQM4OzvL
LKekpIQTJ05gw4YNsLa2ZmVr4vP5WLRoEYqLi9Hf34/u7m6Ul5djz549EvHH
TE1N8erVK5SVlWHVqlVQUVEBwzBgGAZ8Ph9z585FUFCQVGM/wzD49ttvUVNT
wzqkQEtLCzk5OayTL3K5XISFhSEhIQEbNmxAVlYWQkJCZOKpaWhoICsrC729
vSguLsaKFSswefJkMTs6OkoNr5g+fTra29tZ22UMDQ0RHByM0aNHY8OGDVi2
bBk2b94MPz8/zJo1S6K6VuR4VV5ejtLSUmRlZcm07xG980wMCgrCuXPnsGnT
JowZM+aflzmU6J195sWLF6yya/58wk6cOIGamhq5wV4aGhrYu3ev+OIpLy+H
u7u73M0oYl1dXYwZMwbW1tZyjeZcLhdmZmZwcnKCk5MTHB0dYWRkJNVwyTAM
5syZg/j4eNy6dQuBgYGYN28ea3RWBQUFHD16FOvWrRtUUK2on3Z2dvD19cX5
8+dx5swZmaCNBgYGiI+PH5RwQPTOkSApKQmnTp3CuHHj5OrUuVwuJk6ciODg
YOzZswe2trasvs3DwwMFBQWs+8cwDLS1tWFvb4/Ro0dj+PDhrDzXFBUVsWLF
Cpw7d06up5YkNjc3x+nTp2VeVvr6+jhw4AB27doFd3d3jBo1alDtCAQCODg4
sAan/bXMMAx8fX0RHx8/KNTywbYxYsQI8d4yMTGRupb09PRw+PBhqc4rDMPA
xsZGZmiGs7MzQkNDWX/PkCFDcPLkyUHbgLy9vbFr1y5MnTpV7pmkqamJyMhI
cXr5wez5devWITExcVAxM7a2trC0tISVlZU4JmvkyJFiZyxJ9aZOnYo7d+7g
0KFDgwJwZsO/yaUzfvx4BAYGDgotleid5HP69GmZXhE/L29tbQ0HBwex98dv
ORj/CTxx4kSEhoYOKlDuX8kLFizAhg0bfp/b/wU2MzPD3bt35aIP/86/869h
affKvwzwU09PT6yfldXm7/TbkKamJvF4PInOEL/TfzZZWlrSZ599RpcuXfp9
L/5O/zSCFMDPX3zpKCoqEgBWRrjf6dcRn88nIpKaCfF3+ucTh8MRe2GxdURg
Szwej/r7+/9lFwDDMPT5559TS0uLRE8yaaSgoEADAwODXoc6OjpkY2ND6enp
v6/h/yCSdukMGmWax+PR3LlzKSMjg7777ju5CM48Ho90dHRowoQJZGdnR/r6
+uJDlA3Z2dmRk5PTYLtJSkpKNGHCBHGaXnnEMAyNGjWKpk+fLtdFk8vlkoWF
BXl4eNDGjRtpwYIFpK6uPqj+WVtbk4ODg9xy9vb2dPDgQQoNDaWJEycO6nsM
DQ3J29ubXFxcWKXf/nl9tq7FItLQ0JDpKi2JOBwO2drakpubG6mpqQ2qLhsa
NWoUeXt707x580hTU/MX/QbDMDR79mzKy8ujbdu2kUAg+M365+TkRGFhYfTV
V1+RsbEx6/lVVFQkW1tbWrNmDdnZ2Q2qTWtrawoLC6NPPvmEdR0ej0dffvkl
nT17dlAoxqNGjaJz587Rf/3Xf/3ml/X7xOFwxHvSzs5O7h5mGOYXzSOXy6XR
o0fTlClTJKZwl0V8Pp9UVVXlIoczDEPq6uo0YcIEGjduHBkbG8s9Z0Wko6ND
c+bMofnz55OxsTFrRHA+n/+LxoNhGBIKhTRv3jxycnJid2YMxqbD5/OxatUq
vHr1CgUFBXIReFVVVfHtt9+KM1L6+/vj3Llz2LdvHysgQw6Hg+joaNYwFe+z
i4sLrl+/zio1MdE7XLni4mJUVlbC1dVVqq6bx+NhzZo1KC8vx8uXL5GVlYWG
hgacP3+eNVaWQCBAamoqdu3aJbOcoaEhvvvuO5iYmMDCwgIBAQFiUD9ZLAJm
TU9Px6NHj9DY2Ihly5ax1t+rq6vj66+/hr+//6AAJydOnDgo7CtlZWW4u7uj
pqYGXV1d8Pf3l7oOVFVVYWxsjGnTpmHhwoXw9vbGrFmz5KJAGBkZYdWqVcjN
zUVycvKgE++JMiRWVlaiv78flZWVMiPKRQZdV1dXbNmyRWbgoAhhITg4GDEx
McjNzYWXl5dc5w0jIyMcOXIEp0+fxv79+3Hjxg3WBnQ1NTXEx8cjKytr0Fk9
Fy1ahPb2diQkJLBylFBWVkZkZCRCQkL+v/auPSjK63w/e9+FZWGBVVCQrSCg
4g0I0qoxmQiSKWqoiiWrKBFTTcAbVrCCUjBUoVCEsSAh8Ya2Erzi4BXUIFHE
QeOGiyAoIoaLBGEFBIT394ezO1LY3Y+0k/Y3s8/M+UP8Dt85h3POd97zvs/z
6m0fj8fTZAw2MzMjmUymKbsH4HkAABZKSURBVPrWMJvNpsDAQKqtraWenh6q
r6/XS4x0dnamsLAwTYQcEzVrR0dH+uCDDygwMJAWLlxIvr6+5Ofnx3gc3d3d
6dKlS3qDF0aNGkU3b96k9vZ2amtro+bmZoqLi2Pkn128eDE9efKEampqNCm1
mfQtICCAUlNTGak/AG+CPz766CPaunUr5ebmkkqlorKyskFr4z+iSBASEkIq
lYry8vJ0RkSoi4uLC6WkpNDYsWM10RQSiYQUCgXt2LFDb8fYbDadPn2a8vPz
RywVHxISQtnZ2YzqOTo6UmFhIeXk5NCCBQsGaWH9a7G2tqaysjLKzMykyZMn
k6mpKRUUFJBSqWScWtbe3p4qKyv1pkK2tbUdFIrt4OBAMTExehe8paUllZeX
k1KppKCgIMrOzqbi4mJGH26ZTEaZmZn06tUrOnbsGDk4ODDaYFgsFm3ZsoUR
Cxp4w4Y/cuQIvXjxgsrLy6mjo4O+/vrrIc9xuVwKCgqib7/9lmpra6m+vp4e
PnxIdXV11N7eThkZGYxSao8bN45ycnLo9OnTjOXyeTwe+fn5UUtLC6lUKrpz
545WdXAWi0USiYRCQkLo8ePH1NXVRQUFBXoVx/l8PnG5XBKJRLRo0SK6du0a
ffjhh1rXlkAgoL/97W/0+9//noyMjGjevHmUk5PDaLNgsVgUEhJCT548oXff
fXdE62nq1KlUVFREKpWKKioq9Ka7ZrPZpFAo6NSpUzqjztTFz8+PysvL6eLF
i1RaWkoNDQ1UX19PLS0tlJGRoTM03t7enqqqqqijo4Pu3r1LMTExVFhYqHN/
CggI0ChLe3h40Pbt23W2TyqV0t///ndNZKI6mvLKlSt6RXgFAgFZWVmRo6Mj
hYWFUWFhoc5w92XLllF3dzfdvn2bNm3aRH/5y18Yp00XiUQaCTAfHx9qaGig
xYsX660XHBxMlZWVjMLwORwO7d27l7q6uqizs5Pq6uqorq6O9u/fP2i/1fZd
YUwOtbe3x9atW3Hv3j1ERERgwoQJ8PLywrNnz3D16lV0dXUNqVNZWYmoqCi0
t7drftbV1QW5XI4JEybozVhnamqKMWPG4MaNG3pZv2+Dw+Hgt7/9LQ4fPqy3
nkAgQFRUFPLz85GYmIienh4sXboUjo6OKC8vH/L8y5cv8eTJE3z55ZcoKyvD
2LFjMWbMGNTV1TFSJADeXJm1tbWhoqJC53P19fWD/t3S0gILCwuND0Abent7
UVNTg2nTpuHly5f47rvvMG/ePIwePVong18qlSI9PR0LFiwAm83G7NmzcerU
KWRnZyM5ORkqlUprXQ6HA1tb2yGEWm3w9fXFnDlzkJqaivz8fPzzn//Eo0eP
hjxnZGSERYsWobq6GjExMWhsbMTTp08xZ84cHDx4ECKRSO91BRHhyZMn+NOf
/oTc3Fy4ubnpJcwKhUJs3LgRW7ZsgUAgQEJCgibZ308//TToWRaLhaVLlyI0
NBTTp0/XEAgfPHgwaO4PBzVx7/Xr1zhz5gyampoQExOD+/fvD0v8fP36NZKT
k/Hjjz9i3Lhx+PTTT3Hx4kWdBEA1PDw8sHnzZhw6dAhFRUV6n1dj+vTpyMjI
QH19PX744Qd89NFH+NWvfoXvv/9eax2ZTIaFCxciOTkZT58+1fuOy5cvw8LC
AiqVCl1dXWCxWJgxYwY2b96Mzs5OraoLEokE8fHxMDU1xcqVK1FWVgZHR0f4
+/tDJpMNG0jDZrPxzjvvoKSkBAAwceJEvfuEhYUFysvLUVNTg/7+fgiFQixf
vhwuLi4a4udwe5mJiQni4uLA4/FQVFSE9vZ2PH36FO7u7kPWN/Bm3vn7+0Mg
EODu3bsoLCyEm5sb3nnnHRCRXnJ9d3e3xk83ZcoUcDgcRteaAoEAY8aMwfTp
04dt19vgcDiwsrJCf38/srKyUFpaitLSUjx48IDZPs3U0omMjKTXr19TREQE
5eTk0PPnz+nhw4fU3NxM6enpjE4zEomEoqKi6LvvvmOkU2ZlZUWVlZWUmJg4
olOZk5MTlZaW6iU6AaDJkyfT9evXB2l/xcXFadVdY7FY5OXlpeHmhIWF0YsX
Lyg0NJTx9VVERARduHBhxBwJExMTSk9PZ8RZsra2Jnd3d5o/fz7l5eVRd3e3
zutQqVRKX331FfX09NCdO3coOjqafHx8NOSxiIgInea9hYUFnThxgvE1qJWV
FTk4OBCXy6Vt27ZRS0uLVul8mUymmV8mJia0cOFCUiqVjOX2eTweGRkZkVgs
pkuXLunVvVNbOO3t7aRSqSgyMpJmzZpFdXV1dObMmSHjrxZ+7O/vp/7+flKp
VNTf309paWkjDklWk3qjo6N1PqdQKKi+vp7u3r1Lly5donnz5ul8l5GREZ05
c4aqqqo060IoFJKNjY1OK8nGxoZu3rxJdXV1mvnQ1NSkcy6x2WyKjIyk48eP
k1gspvHjx5Ofnx+5ubnpvDp8u/12dnZUUlJC5eXlOtexWrk5ICBAMz+Dg4Pp
2bNn5OTkNGwdY2NjOnz4sIaXsn79er35dKZNm0bnz5+n5cuXk729vYZLeOLE
CZ03HMHBwXTkyJFBV4RBQUEUHx+vdV20tLTQwMAAdXV1adKtdHV1UXNzM3l5
eTGaR1OnTqXi4mLav38/I8JxREQENTY2DtEPHK6YmJhQfn4+9fX1kUqlou7u
brp169YQLb5/29IZN24c+vr64ODggPfeew8bNmzA9evXMXPmTISEhGDFihXY
vXv3sF97DocDT09PfPLJJ+Dz+fjkk0+GlX74V0gkEpiamjJtogaLFi1CTU0N
I6kZLy8vlJSUaKRAWCwWhEKhVocYEeHy5csA3ozJihUrUF1djaysLEbRRxwO
B7/+9a9RXFyM7u7uEfTqzQmtvb2d0cmlra0Na9aswWeffQYzMzNwOBz4+/uj
urp6iFWqtvYCAgLw6tUrHD16FGlpaXj16hVu3bqFiRMn6k2d7ODggJ9++gkv
X77U2zYulws7OzsEBASgv78f/v7+OHfunFbLTy1/wuVysWvXLqxZswYCgQCd
nZ1Yu3YtwsLCtI4ll8tFREQEXF1dER0djfb2dp2nMR6PB19fX6SlpYHNZiM+
Ph7nzp1DYmIiJBIJUlNTh9Tv7e3FxYsXMXHiRJSUlODkyZNISkqCSCQCm83W
+fcyMTGBr68vbGxscOzYMchkMjg6OsLKykrnTUBZWRnCwsKQn58PV1dX7Nix
A42Njfjhhx+Gfd7b2xu/+c1vsGnTJjx58gSenp4ICwuDm5sbsrOzsXPnziGR
qGw2G+Hh4XB0dERQUBCuX78OhUKBpqYmnXpgIpEIrq6uOHv2LFavXo3Vq1ej
qakJ/f392Lhxo9a1r+6rsbExYmNjYWNjg3Xr1ml9l0wmQ2BgIIqLi3H27FmN
NaRSqdDf3691DRsZGUEul8PIyAguLi6Qy+V6Jb2+//57pKSkYP/+/eByueDx
eDh9+jS2bdumlZbAZrMxd+5cfPnll4Okcrhcrtao397eXjQ3N0MqlaKmpgbX
rl3DuXPnMG3aNMTExMDR0VGz/2jr24IFCxAREQGlUomIiAi9+wybzcaoUaNg
bm6usdJ14f3334eHhwdaWlqQlJQECwsLbNq0CT4+Poz03hhbOuHh4fTixQvK
ycmh6upqzemDw+FQYmKi1tOjsbExrV+/nkpKSighIYGcnZ0ZO9wVCgX19vbS
zp07GZ8UjY2N6dtvv2WUD4bD4VBubu4g1q9YLKa8vDy9Jwo2m007duyg3t5e
ioqKIqlUShKJRO8pwd7eXpPYiWmfhEKhJgnU9u3b9ToU1YnBlEolbdiwgerr
6+n27dtUXl5Of/jDH4b4aN577z3q6Oigo0ePkre3N7m7u5NUKqX58+dTbm4u
3b59W6dFoT4prVmzRm9fRCIRffHFF6RUKqmhoYEGBgaotbWVcaZSJycnmjdv
Hnl5eVFSUhK1trbqzHcjl8uptraWCgoK6MKFC1RRUaHVnyEQCGjr1q3U0tJC
ra2tFBcXR2vXrqXq6mrq6+vTGZjC4XDI2dmZzMzMaPbs2dTe3k4FBQU6/Udz
586l8+fP0/Hjxyk+Pp5mzZpF6enppFKpKC0tjfH8UMuZJCcnD/v/pqamdOXK
FTpy5AgZGRnRokWL6PHjx3Tr1i16/Pgx3bt3b9iABw8PD2psbKRDhw6RiYmJ
Jn3Ahg0bdLZHIBDQsWPHqLq6mqqqqig0NJQsLS0pJSVFb9plPp9Pe/bsoYaG
BgoICNDpT5w+fTo1NzcPUUhxdXUlpVKplWFvbGxM6enplJeXR4WFhVRfX88o
RcbcuXPp0aNH1NbWRnv37tWbBE8oFNKBAwcGWUJisZiOHj2q9aaHxWLRwoUL
ac+ePTRu3DjNz0NCQqinp4eWLFkyZKzXrVtH2dnZlJqaSpcvXyaVSkVJSUl6
/YnqwuVyKSMjg1QqlV5ZJOCNpdbd3U0pKSnE4XDI0dGRnj17RklJSYP2v3/b
0jlx4gRCQkIwc+ZMmJubIyoqClu3boW3tzfs7e2RkJAwpI61tTWio6Px8ccf
o6mpCVOmTEFCQgJaW1tx48YNHD9+XKefoKGhAUSEe/fuMW0mbG1tYW1tzciS
IiJ0dXVpTlgsFgvz5s0Di8VCcXGx1nocDgdubm7w8/MDm83GqlWrsHjxYvT0
9KCwsBCHDx+GUqkc9pS6ZMkS3L9/H2VlZXrbx2KxEBgYCDc3N/T09GDChAk6
xTHVGD16NBQKBSIiIuDk5AQ+n4/169eDiDB79mzweLxBp28fHx+IRCJYWlpi
yZIl6OjogIODA6ytrXHjxg1s27ZN6wkaAMRiMTw8PBAZGamzXWw2G2vXrsXY
sWMRFxeHTZs2wcrKCmKxGJ6enrhz547evj148AAPHjwAj8fD7Nmz0dPTo9NP
xefzQURIS0vDnj17cPfuXdy+fXvYZ2fOnInw8HCYmZmho6MD3t7emDx5Mrhc
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
" /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Construting datasets ...</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Defining neural networks ...</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Defining optimizer and cost function ...</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Starting training ...</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 1 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  1/88 | Batch: 128/700 | Cost: 0.676701</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  1/88 | Batch: 256/700 | Cost: 0.670129</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  1/88 | Batch: 384/700 | Cost: 0.664933</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  1/88 | Batch: 512/700 | Cost: 0.661664</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  1/88 | Batch: 640/700 | Cost: 0.663557</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 2 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  2/88 | Batch: 128/700 | Cost: 0.662617</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  2/88 | Batch: 256/700 | Cost: 0.662023</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  2/88 | Batch: 384/700 | Cost: 0.661249</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  2/88 | Batch: 512/700 | Cost: 0.659035</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  2/88 | Batch: 640/700 | Cost: 0.659464</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 3 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  3/88 | Batch: 128/700 | Cost: 0.656966</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  3/88 | Batch: 256/700 | Cost: 0.658323</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  3/88 | Batch: 384/700 | Cost: 0.657182</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  3/88 | Batch: 512/700 | Cost: 0.659456</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  3/88 | Batch: 640/700 | Cost: 0.658706</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 4 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  4/88 | Batch: 128/700 | Cost: 0.658838</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  4/88 | Batch: 256/700 | Cost: 0.657326</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  4/88 | Batch: 384/700 | Cost: 0.657664</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  4/88 | Batch: 512/700 | Cost: 0.659858</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  4/88 | Batch: 640/700 | Cost: 0.657197</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 5 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  5/88 | Batch: 128/700 | Cost: 0.657947</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  5/88 | Batch: 256/700 | Cost: 0.659117</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  5/88 | Batch: 384/700 | Cost: 0.657783</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  5/88 | Batch: 512/700 | Cost: 0.658427</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  5/88 | Batch: 640/700 | Cost: 0.658063</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 6 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  6/88 | Batch: 128/700 | Cost: 0.657641</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  6/88 | Batch: 256/700 | Cost: 0.656912</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  6/88 | Batch: 384/700 | Cost: 0.657368</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  6/88 | Batch: 512/700 | Cost: 0.659303</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  6/88 | Batch: 640/700 | Cost: 0.658205</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 7 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  7/88 | Batch: 128/700 | Cost: 0.655346</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  7/88 | Batch: 256/700 | Cost: 0.658861</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  7/88 | Batch: 384/700 | Cost: 0.656757</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  7/88 | Batch: 512/700 | Cost: 0.658256</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  7/88 | Batch: 640/700 | Cost: 0.658176</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 8 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  8/88 | Batch: 128/700 | Cost: 0.655609</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  8/88 | Batch: 256/700 | Cost: 0.656722</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  8/88 | Batch: 384/700 | Cost: 0.656618</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  8/88 | Batch: 512/700 | Cost: 0.657232</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  8/88 | Batch: 640/700 | Cost: 0.656805</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655383, Training accuracy: 97.98, Validation accuracy: 97.51</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 9 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  9/88 | Batch: 128/700 | Cost: 0.656998</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  9/88 | Batch: 256/700 | Cost: 0.656764</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  9/88 | Batch: 384/700 | Cost: 0.658776</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  9/88 | Batch: 512/700 | Cost: 0.656042</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch:  9/88 | Batch: 640/700 | Cost: 0.656336</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 10 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 10/88 | Batch: 128/700 | Cost: 0.656183</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 10/88 | Batch: 256/700 | Cost: 0.657768</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 10/88 | Batch: 384/700 | Cost: 0.655452</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 10/88 | Batch: 512/700 | Cost: 0.658559</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 10/88 | Batch: 640/700 | Cost: 0.656760</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 11 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 11/88 | Batch: 128/700 | Cost: 0.658124</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 11/88 | Batch: 256/700 | Cost: 0.657016</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 11/88 | Batch: 384/700 | Cost: 0.656468</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 11/88 | Batch: 512/700 | Cost: 0.657117</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 11/88 | Batch: 640/700 | Cost: 0.656739</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 12 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 12/88 | Batch: 128/700 | Cost: 0.656247</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 12/88 | Batch: 256/700 | Cost: 0.656758</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 12/88 | Batch: 384/700 | Cost: 0.655410</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 12/88 | Batch: 512/700 | Cost: 0.656733</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 12/88 | Batch: 640/700 | Cost: 0.658109</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 13 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 13/88 | Batch: 128/700 | Cost: 0.655228</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 13/88 | Batch: 256/700 | Cost: 0.657007</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 13/88 | Batch: 384/700 | Cost: 0.655182</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 13/88 | Batch: 512/700 | Cost: 0.656848</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 13/88 | Batch: 640/700 | Cost: 0.656587</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 14 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 14/88 | Batch: 128/700 | Cost: 0.656211</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 14/88 | Batch: 256/700 | Cost: 0.656756</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 14/88 | Batch: 384/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 14/88 | Batch: 512/700 | Cost: 0.657861</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 14/88 | Batch: 640/700 | Cost: 0.656843</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 15 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 15/88 | Batch: 128/700 | Cost: 0.655258</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 15/88 | Batch: 256/700 | Cost: 0.656734</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 15/88 | Batch: 384/700 | Cost: 0.655682</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 15/88 | Batch: 512/700 | Cost: 0.657177</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 15/88 | Batch: 640/700 | Cost: 0.657033</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 16 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 16/88 | Batch: 128/700 | Cost: 0.657406</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 16/88 | Batch: 256/700 | Cost: 0.657553</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 16/88 | Batch: 384/700 | Cost: 0.655170</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 16/88 | Batch: 512/700 | Cost: 0.656774</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 16/88 | Batch: 640/700 | Cost: 0.656668</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655191, Training accuracy: 99.05, Validation accuracy: 98.42</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 17 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 17/88 | Batch: 128/700 | Cost: 0.655225</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 17/88 | Batch: 256/700 | Cost: 0.658285</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 17/88 | Batch: 384/700 | Cost: 0.655230</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 17/88 | Batch: 512/700 | Cost: 0.656942</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 17/88 | Batch: 640/700 | Cost: 0.655755</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 18 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 18/88 | Batch: 128/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 18/88 | Batch: 256/700 | Cost: 0.656726</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 18/88 | Batch: 384/700 | Cost: 0.655164</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 18/88 | Batch: 512/700 | Cost: 0.656782</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 18/88 | Batch: 640/700 | Cost: 0.656509</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 19 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 19/88 | Batch: 128/700 | Cost: 0.655165</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 19/88 | Batch: 256/700 | Cost: 0.657053</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 19/88 | Batch: 384/700 | Cost: 0.655167</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 19/88 | Batch: 512/700 | Cost: 0.656688</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 19/88 | Batch: 640/700 | Cost: 0.656567</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 20 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 20/88 | Batch: 128/700 | Cost: 0.655198</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 20/88 | Batch: 256/700 | Cost: 0.656715</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 20/88 | Batch: 384/700 | Cost: 0.655181</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 20/88 | Batch: 512/700 | Cost: 0.656704</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 20/88 | Batch: 640/700 | Cost: 0.655502</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 21 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 21/88 | Batch: 128/700 | Cost: 0.655215</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 21/88 | Batch: 256/700 | Cost: 0.656776</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 21/88 | Batch: 384/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 21/88 | Batch: 512/700 | Cost: 0.655890</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 21/88 | Batch: 640/700 | Cost: 0.656553</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 22 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 22/88 | Batch: 128/700 | Cost: 0.655260</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 22/88 | Batch: 256/700 | Cost: 0.656714</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 22/88 | Batch: 384/700 | Cost: 0.655176</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 22/88 | Batch: 512/700 | Cost: 0.656509</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 22/88 | Batch: 640/700 | Cost: 0.655172</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 23 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 23/88 | Batch: 128/700 | Cost: 0.655229</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 23/88 | Batch: 256/700 | Cost: 0.656080</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 23/88 | Batch: 384/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 23/88 | Batch: 512/700 | Cost: 0.655933</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 23/88 | Batch: 640/700 | Cost: 0.655541</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 24 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 24/88 | Batch: 128/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 24/88 | Batch: 256/700 | Cost: 0.655341</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 24/88 | Batch: 384/700 | Cost: 0.655364</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 24/88 | Batch: 512/700 | Cost: 0.655170</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 24/88 | Batch: 640/700 | Cost: 0.655558</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655206, Training accuracy: 99.52, Validation accuracy: 98.79</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 25 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 25/88 | Batch: 128/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 25/88 | Batch: 256/700 | Cost: 0.655393</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 25/88 | Batch: 384/700 | Cost: 0.655292</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 25/88 | Batch: 512/700 | Cost: 0.656108</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 25/88 | Batch: 640/700 | Cost: 0.655298</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 26 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 26/88 | Batch: 128/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 26/88 | Batch: 256/700 | Cost: 0.656238</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 26/88 | Batch: 384/700 | Cost: 0.655215</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 26/88 | Batch: 512/700 | Cost: 0.655236</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 26/88 | Batch: 640/700 | Cost: 0.655163</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 27 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 27/88 | Batch: 128/700 | Cost: 0.655169</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 27/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 27/88 | Batch: 384/700 | Cost: 0.655191</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 27/88 | Batch: 512/700 | Cost: 0.655188</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 27/88 | Batch: 640/700 | Cost: 0.655589</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 28 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 28/88 | Batch: 128/700 | Cost: 0.655167</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 28/88 | Batch: 256/700 | Cost: 0.656033</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 28/88 | Batch: 384/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 28/88 | Batch: 512/700 | Cost: 0.655264</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 28/88 | Batch: 640/700 | Cost: 0.655216</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 29 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 29/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 29/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 29/88 | Batch: 384/700 | Cost: 0.655558</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 29/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 29/88 | Batch: 640/700 | Cost: 0.655302</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 30 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 30/88 | Batch: 128/700 | Cost: 0.655167</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 30/88 | Batch: 256/700 | Cost: 0.656682</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 30/88 | Batch: 384/700 | Cost: 0.655192</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 30/88 | Batch: 512/700 | Cost: 0.655193</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 30/88 | Batch: 640/700 | Cost: 0.655179</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 31 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 31/88 | Batch: 128/700 | Cost: 0.656356</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 31/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 31/88 | Batch: 384/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 31/88 | Batch: 512/700 | Cost: 0.655201</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 31/88 | Batch: 640/700 | Cost: 0.655242</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 32 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 32/88 | Batch: 128/700 | Cost: 0.655167</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 32/88 | Batch: 256/700 | Cost: 0.656473</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 32/88 | Batch: 384/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 32/88 | Batch: 512/700 | Cost: 0.655163</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 32/88 | Batch: 640/700 | Cost: 0.655257</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655241, Training accuracy: 99.54, Validation accuracy: 98.71</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 33 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 33/88 | Batch: 128/700 | Cost: 0.655171</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 33/88 | Batch: 256/700 | Cost: 0.655165</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 33/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 33/88 | Batch: 512/700 | Cost: 0.656383</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 33/88 | Batch: 640/700 | Cost: 0.655423</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 34 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 34/88 | Batch: 128/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 34/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 34/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 34/88 | Batch: 512/700 | Cost: 0.655998</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 34/88 | Batch: 640/700 | Cost: 0.655209</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 35 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 35/88 | Batch: 128/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 35/88 | Batch: 256/700 | Cost: 0.655163</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 35/88 | Batch: 384/700 | Cost: 0.655192</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 35/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 35/88 | Batch: 640/700 | Cost: 0.655173</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 36 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 36/88 | Batch: 128/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 36/88 | Batch: 256/700 | Cost: 0.655966</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 36/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 36/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 36/88 | Batch: 640/700 | Cost: 0.655167</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 37 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 37/88 | Batch: 128/700 | Cost: 0.655190</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 37/88 | Batch: 256/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 37/88 | Batch: 384/700 | Cost: 0.655236</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 37/88 | Batch: 512/700 | Cost: 0.655182</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 37/88 | Batch: 640/700 | Cost: 0.655347</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 38 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 38/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 38/88 | Batch: 256/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 38/88 | Batch: 384/700 | Cost: 0.655301</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 38/88 | Batch: 512/700 | Cost: 0.655169</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 38/88 | Batch: 640/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 39 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 39/88 | Batch: 128/700 | Cost: 0.655172</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 39/88 | Batch: 256/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 39/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 39/88 | Batch: 512/700 | Cost: 0.655208</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 39/88 | Batch: 640/700 | Cost: 0.655164</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 40 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 40/88 | Batch: 128/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 40/88 | Batch: 256/700 | Cost: 0.655203</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 40/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 40/88 | Batch: 512/700 | Cost: 0.655165</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 40/88 | Batch: 640/700 | Cost: 0.655177</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655159, Training accuracy: 99.73, Validation accuracy: 98.86</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 41 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 41/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 41/88 | Batch: 256/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 41/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 41/88 | Batch: 512/700 | Cost: 0.655164</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 41/88 | Batch: 640/700 | Cost: 0.655169</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 42 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 42/88 | Batch: 128/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 42/88 | Batch: 256/700 | Cost: 0.655250</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 42/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 42/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 42/88 | Batch: 640/700 | Cost: 0.655268</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 43 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 43/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 43/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 43/88 | Batch: 384/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 43/88 | Batch: 512/700 | Cost: 0.655185</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 43/88 | Batch: 640/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 44 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 44/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 44/88 | Batch: 256/700 | Cost: 0.655173</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 44/88 | Batch: 384/700 | Cost: 0.655163</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 44/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 44/88 | Batch: 640/700 | Cost: 0.655180</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 45 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 45/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 45/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 45/88 | Batch: 384/700 | Cost: 0.655349</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 45/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 45/88 | Batch: 640/700 | Cost: 0.655295</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 46 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 46/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 46/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 46/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 46/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 46/88 | Batch: 640/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 47 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 47/88 | Batch: 128/700 | Cost: 0.655169</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 47/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 47/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 47/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 47/88 | Batch: 640/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 48 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 48/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 48/88 | Batch: 256/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 48/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 48/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 48/88 | Batch: 640/700 | Cost: 0.655165</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655163, Training accuracy: 99.68, Validation accuracy: 98.68</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 49 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 49/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 49/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 49/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 49/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 49/88 | Batch: 640/700 | Cost: 0.655177</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 50 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 50/88 | Batch: 128/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 50/88 | Batch: 256/700 | Cost: 0.655216</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 50/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 50/88 | Batch: 512/700 | Cost: 0.655169</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 50/88 | Batch: 640/700 | Cost: 0.655268</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 51 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 51/88 | Batch: 128/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 51/88 | Batch: 256/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 51/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 51/88 | Batch: 512/700 | Cost: 0.655165</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 51/88 | Batch: 640/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 52 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 52/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 52/88 | Batch: 256/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 52/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 52/88 | Batch: 512/700 | Cost: 0.655167</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 52/88 | Batch: 640/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 53 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 53/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 53/88 | Batch: 256/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 53/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 53/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 53/88 | Batch: 640/700 | Cost: 0.655163</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 54 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 54/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 54/88 | Batch: 256/700 | Cost: 0.655206</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 54/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 54/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 54/88 | Batch: 640/700 | Cost: 0.655170</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 55 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 55/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 55/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 55/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 55/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 55/88 | Batch: 640/700 | Cost: 0.655165</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 56 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 56/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 56/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 56/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 56/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 56/88 | Batch: 640/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655159, Training accuracy: 99.78, Validation accuracy: 98.92</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 57 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 57/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 57/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 57/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 57/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 57/88 | Batch: 640/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 58 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 58/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 58/88 | Batch: 256/700 | Cost: 0.655382</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 58/88 | Batch: 384/700 | Cost: 0.655170</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 58/88 | Batch: 512/700 | Cost: 0.655165</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 58/88 | Batch: 640/700 | Cost: 0.655215</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 59 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 59/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 59/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 59/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 59/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 59/88 | Batch: 640/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 60 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 60/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 60/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 60/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 60/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 60/88 | Batch: 640/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 61 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 61/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 61/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 61/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 61/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 61/88 | Batch: 640/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 62 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 62/88 | Batch: 128/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 62/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 62/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 62/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 62/88 | Batch: 640/700 | Cost: 0.655163</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 63 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 63/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 63/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 63/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 63/88 | Batch: 512/700 | Cost: 0.655166</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 63/88 | Batch: 640/700 | Cost: 0.655164</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 64 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 64/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 64/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 64/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 64/88 | Batch: 512/700 | Cost: 0.655169</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 64/88 | Batch: 640/700 | Cost: 0.655163</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655161, Training accuracy: 99.79, Validation accuracy: 99.0</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 65 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 65/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 65/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 65/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 65/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 65/88 | Batch: 640/700 | Cost: 0.655165</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 66 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 66/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 66/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 66/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 66/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 66/88 | Batch: 640/700 | Cost: 0.655164</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 67 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 67/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 67/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 67/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 67/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 67/88 | Batch: 640/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 68 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 68/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 68/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 68/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 68/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 68/88 | Batch: 640/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 69 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 69/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 69/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 69/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 69/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 69/88 | Batch: 640/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 70 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 70/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 70/88 | Batch: 256/700 | Cost: 0.655999</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 70/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 70/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 70/88 | Batch: 640/700 | Cost: 0.655170</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 71 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 71/88 | Batch: 128/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 71/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 71/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 71/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 71/88 | Batch: 640/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 72 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 72/88 | Batch: 128/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 72/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 72/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 72/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 72/88 | Batch: 640/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655159, Training accuracy: 99.79, Validation accuracy: 98.96</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 73 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 73/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 73/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 73/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 73/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 73/88 | Batch: 640/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 74 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 74/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 74/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 74/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 74/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 74/88 | Batch: 640/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 75 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 75/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 75/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 75/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 75/88 | Batch: 512/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 75/88 | Batch: 640/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 76 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 76/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 76/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 76/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 76/88 | Batch: 512/700 | Cost: 0.655163</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 76/88 | Batch: 640/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 77 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 77/88 | Batch: 128/700 | Cost: 0.655333</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 77/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 77/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 77/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 77/88 | Batch: 640/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 78 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 78/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 78/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 78/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 78/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 78/88 | Batch: 640/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 79 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 79/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 79/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 79/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 79/88 | Batch: 512/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 79/88 | Batch: 640/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 80 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 80/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 80/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 80/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 80/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 80/88 | Batch: 640/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655159, Training accuracy: 99.79, Validation accuracy: 99.03</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 81 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 81/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 81/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 81/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 81/88 | Batch: 512/700 | Cost: 0.655167</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 81/88 | Batch: 640/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 82 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 82/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 82/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 82/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 82/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 82/88 | Batch: 640/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 83 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 83/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 83/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 83/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 83/88 | Batch: 512/700 | Cost: 0.655162</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 83/88 | Batch: 640/700 | Cost: 0.655161</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 84 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 84/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 84/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 84/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 84/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 84/88 | Batch: 640/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 85 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 85/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 85/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 85/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 85/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 85/88 | Batch: 640/700 | Cost: 0.655160</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 86 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 86/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 86/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 86/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 86/88 | Batch: 512/700 | Cost: 0.655533</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 86/88 | Batch: 640/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 87 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 87/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 87/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 87/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 87/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 87/88 | Batch: 640/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch 88 -------------------------------------------------------------</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 88/88 | Batch: 128/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 88/88 | Batch: 256/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 88/88 | Batch: 384/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 88/88 | Batch: 512/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Epoch: 88/88 | Batch: 640/700 | Cost: 0.655159</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">&gt;&gt; Cost: 0.655159, Training accuracy: 99.79, Validation accuracy: 98.98</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Finishing training ...</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Training time: 163.76752829551697</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><img src="data:image/png;base64,
iVBORw0KGgoAAAANSUhEUgAAAZUAAAEWCAYAAACufwpNAAAACXBIWXMAAAsT
AAALEwEAmpwYAAAAO3pUWHRTb2Z0d2FyZQAACJnzTSwpyMkvyclMUihLLSrO
zM8z1jPVM9BRyCgpKSi20tfPhSvQyy9K1wcAqSMRJwjwuEgAACAASURBVHic
7d15nFxVmf/xz5dOgCZEghLUBEKCBhRUgrSgIhpANhmFcRxFR8GVgRH3Hwoz
LrgNCCquDCKbGwI6SYw6JqCYxAWEjoGsBEKAkIQlQQIYIiTh+f1xToWbSlX3
7U5Vd1f6+369+tV3v8+9tTx1zzn3XEUEZmZmjbBdfwdgZmbbDicVMzNrGCcV
MzNrGCcVMzNrGCcVMzNrGCcVMzNrGCeVfiJprKSQNKS/Y+lrku6R9Pp+2O+X
JK2W9EDJ5c+R9ONmx7W1JL1b0h9LLnulpC+VXHar3qOSxkj6u6S23qzfi/31
y/uqHkm/kXRKo5cd6JxU+shAe8N3ZVtMeJL2BD4B7BcRz6sxf6Kk5X0f2bYr
IpZFxM4RsbHR2+5Jcuzl9kPSC7dmGxFxXET8oNHLDnROKjZY7AU8HBEP9Xcg
g8G29IOklm39+LaGk0ofkPQjYAzwy1wc8MnC7H+TtCwXy/xXYZ3tJJ0l6S5J
D0u6VtKzu9jHCZJulfRYXufYPH2UpKmS/iZpiaQPFNY5WFJnXudBSV/Ps2bl
/2tyvK+q2tcoSeuK8Ug6MB/DUEkvkHRDjnu1pJ9IGlEn7s1+cVZfMeR9/a+k
VZLulvThLs7BLpJ+mJe9V9Kn83l8PXA9MCofz5VV6w0DflOY/3dJo/Ls7fM2
H5e0QFJHL2O7UtJFuZjj75L+JOl5kr4h6RFJt0s6sLD8iyXNkLQm7/dNhXnP
ya/pY5JuBl5Qta8XSbo+v+aLJb21XlxV67VJ+mp+zZYCx1fN3+xqW4XiwcLV
7fskLQNuqL7izcfzxXzsj0u6TtJuhe2dnF+3hyV9pnp/heVOBf4N+GQ+l78s
zJ4gaa6kRyVdI2nHwnr/lD8jayT9WdLL6pyHyvv/trz9t1Xel5I+pVR8eoWk
XSX9Kr/+j+ThPQrbmSHp/Xn43ZL+mM/vI/n9clwvlx0naVY+h7+V9F0NpGLa
iPBfH/wB9wCvL4yPBQL4PtAOHAA8Cbw4z/8ocBOwB7AD8D3gp3W2fTDwKHAU
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
29uCi7/MzKxhnFTMzKxhnFTMzKxhnFTMzKxhnFTMzKxhBnXrr2aYMmcFF0xf
zMo16xg1op0zj9mXEw/c2kd6mJm1BieVBpoyZwVnT5rHuvXp+Twr1qzj7Enz
AJxYzGxQcPFXA10wffGmhFKxbv1GLpi+uJ8iMjPrW04qDbRyzboeTTcz29Y4
qTTQqBHtPZpuZratcVJpoDOP2Zf2oW2bTWsf2saZx+xbZw0zs22LK+obqFIZ
79ZfZjZYOak02IkHjnYSMbNBy8VfZmbWME4qZmbWME4qZmbWME4qZmbWME1N
KpKOlbRY0hJJZ9VZZqKkWyUtkDSzMP0eSfPyvM7C9AmSbqpMl3Rwnj5W0ro8
/VZJFzfz2MzMbEtNa/0lqQ34LnAUsBy4RdLUiFhYWGYEcBFwbEQsk7R71WYO
j4jVVdPOBz4fEb+R9IY8PjHPuysiJjThcMzMrIRmXqkcDCyJiKUR8RRwNXBC
1TLvACZFxDKAiHioxHYDeFYe3gVY2aB4zcxsKzUzqYwG7iuML8/TivYBdpU0
Q9JsSScX5gVwXZ5+amH6R4ELJN0HfBU4uzBvnKQ5kmZKOqxWUJJOzcVmnatW
rertsZmZWQ3NvPlRNaZFjf0fBBwJtAM3SropIu4ADo2IlblI7HpJt0fELOB0
4GMR8b+S3gpcBrweuB8YExEPSzoImCJp/4h4bLMAIi4BLgHo6OiojsfMzLZC
M69UlgN7Fsb3YMuiquXAtIhYm+tOZgEHAETEyvz/IWAyqTgN4BRgUh7+WWV6
RDwZEQ/n4dnAXaQrITMz6yPNTCq3AOMljZO0PXASMLVqmV8Ah0kaImkn4BBg
kaRhkoYDSBoGHA3Mz+usBF6Xh48A7szLjcyNA5C0NzAeWNq0ozMzsy00rfgr
IjZIOgOYDrQBl0fEAkmn5fkXR8QiSdOAucDTwKURMT8nhcmSKjFeFRHT8qY/
AHxT0hDgH0ClvuW1wBckbQA2AqdFxN+adXxmZrYlRQzeaoWOjo7o7OzsfkEz
M9tE0uyI6Kg1z3fUm5lZwzipmJlZwzipmJlZwzipmJlZwzipmJlZwzipmJlZ
wzipmJlZwzipmJlZw3SZVCS1SfpxXwVjZmatrcukEhEbgZG57y4zM7Mulen7
6x7gT5KmAmsrEyPi680KyszMWlOZpLIy/20HDG9uOGZm1sq6TSoR8XmA3BV9
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
G5d0rKTFkpZIOqvOMhMl3SppgaSZhen3SJqX53VWrfOhvN0Fks4vTD8772ux
pGPKxNgs/X2lsHLNuh5NNzNrhDI3P14iaVfgM8BUYOc83CVJbcB3gaOA5cAt
kqZGxMLCMiOAi4BjI2KZpOonSh4eEaurtns4cALwsoh4srKOpP2Ak0h1P6OA
30raJyI20g+6ulLoi3qNUSPaWVEjgYwa0d70fZvZ4FXmSuWKiHgkImZGxN4R
sXtEfK/71TgYWBIRS/OTI68mJYOidwCTImIZQEQ8VGK7pwPnRcSTVeucAFwd
ea+FywAAFO5JREFUEU9GxN3AkhxDv+jvK4Uzj9mX9qFtm01rH9rGmcfs2yf7
N7PBqUxSuVvSJZKOlKQebHs0cF9hfHmeVrQPsKukGZJmSzq5MC+A6/L0U6vW
OUzSXyTNlPSKHuwPSadK6pTUuWrVqh4cTs/UuyLoqyuFEw8czblvfimjR7Qj
YPSIds5980vd+svMmqpM8de+wBuBDwKXS/ol6Yrgj92sVysBVT+XZQhwEHAk
0A7cKOmmiLgDODQiVubiresl3R4Rs/I6uwKvBF4BXCtp75L7IyIuAS4B6Ojo
2GJ+o5x5zL6cPWneZkVgfX2lcOKBo51EzKxPlen6fl1EXBsRbwYmAM8iNS3u
znJgz8L4HsDKGstMi4i1ue5kFnBA3u/K/P8hYDLPFGUtJxWZRUTcDDwN7FZy
f33GVwpmNhiVuVJB0uuAtwHHAbdQrpfiW4DxksYBK0iV6O+oWuYXwHckDQG2
Bw4BLpQ0DNguIh7Pw0cDX8jrTCE9z2WGpH3yeqtJjQiukvR1UkX9eODmMsfX
LL5SMLPBpuzjhG8FrgXOjIi1ZTYcERsknQFMB9qAyyNigaTT8vyLI2KRpGnA
XNIVx6URMT8XZ03OVThDgKsiYlre9OWkYrj5wFPAKRERwAJJ1wILSV30f7C/
Wn6ZmQ1WSt/HXSwgPSsiHuujePpUR0dHdHZ2dr+gmZltIml2RHTUmlemTmWb
TChmZtZ4ZZoUm5mZleKkYmZmDdNtUpH0XEmXSfpNHt9P0vuaH5qZmbWaMlcq
V5JacI3K43cAH21WQGZm1rrKJJXdIuJaUpNfImID4Ka6Zma2hTI3P66V9Bxy
lyeSXgk82tSobDN+gqOZtYoySeXjpLvVXyDpT8BI4C1Njco2qTyXpdKHWOW5
LIATi5kNON0mlYj4a+6mZV9Sp42LI2J9N6tZg/T3c1nMzHqiVN9fpM4cx+bl
Xy6JiPhh06KyTfr7uSxmZj1Rpu+vHwEvIPX/VfnJHICTSh/wExzNrJWUuVLp
APaL7joJs6YYCM9lMTMrq0xSmQ88D7i/ybFYDZV6kzKtv9xKzMz6W92kkp/w
GMBwYKGkm4EnK/Mj4k3ND8+g3HNZ3ErMzAaCrq5UvtpnUdhWcysxMxsI6iaV
iJgJIOkrEfGp4jxJX6HcI4Wtj7iVmJkNBGW6aTmqxrTjGh2IbZ16rcHcSszM
+lLdpCLpdEnzgH0lzS383U16/K8NIGcesy/tQ9s2m+ZWYmbW17qqU7kK+A1w
LnBWYfrjEfG3pkZlPdaTVmJmZs3S7TPqt2rj0rHAN4E24NKIOK/GMhOBbwBD
gdUR8bo8/R7gcdINlxsqz0OWdA7wAWBV3sR/RsT/SRoLLAIW5+k3RcRpXcXn
Z9SbmfVcV8+oL9tNS2922gZ8l1Qnsxy4RdLUiFhYWGYEcBFwbEQsk7R71WYO
j4jVNTZ/YUTUap12V0RMaNAhmJlZDzXzccIHA0siYmlEPAVcDZxQtcw7gEkR
sQwgIh5qYjxmZtZkzUwqo4H7CuPL87SifYBdJc2QNFvSyYV5AVyXp59atd4Z
udHA5ZJ2LUwfJ2mOpJmSDqsVlKRTJXVK6ly1alWtRczMrJeaVvxF6ia/WnUF
zhDgIOBIoB24UdJNEXEHcGhErMxFYtdLuj0iZgH/A3wxb+uLwNeA95K6kRkT
EQ9LOgiYImn/iHhsswAiLgEugVSn0qiD7SvuisXMBrJmXqksB/YsjO8BrKyx
zLSIWJvrTmYBBwBExMr8/yFgMqk4jYh4MCI2RsTTwPcL05+MiIfz8GzgLtKV
0Daj0hXLijXrCJ7pimXKnBX9HZqZGdDcpHILMF7SOEnbAyeRniBZ9AvgMElD
JO0EHAIskjRM0nAAScOAo0kdWyLp+YX1/7kwfWRuHICkvYHxwNKmHV0TTZmz
gkPPu4FxZ/2aQ8+7YVPS6KorFjOzgaBpxV8RsUHSGcB0UpPiyyNigaTT8vyL
I2KRpGmkmymfJjU7np+TwmRJlRiviohpedPnS5pAKv66B/j3PP21wBckbSA1
Qz6tFe+n6apjSHfFYmYDXVPvUxnoBuJ9Koeed0PNh3KNzt2t1Jv3p7OOaHps
ZmbQT/epWO90dTVy4dsm9MkDu1qxMUArxmy2LXJSGWC6enxwX3TF0orPZWnF
mM22VU4qA0x3jw8u88CurdGKz2VpxZjNtlVOKgNMf3cM2YqNAVoxZrNtlZPK
ANTsq5GudFX8NlC1Ysxm26pm3qdiLai757LUu4emP7VizGbbKjcpHmBNigeC
ei2pqivEIfXFE6Rmzf3Z4qonMbcPbePcN7/U9S1mvdRVk2InlRZPKn3ZlLbe
PTQVZb+sB0LMvrfHrPd8n8o2qidNabv6Ii/7Jd9dxXdXLa4q+1ixZt2mq5vu
Yu7u2LcmZlfimzWHk0oLK9uUtqvkA3SZmIpf3ttJbOzmynbFmnWMO+vX7NI+
FAnWPLGeXdqHsvapDazfmNat3kIx5jLJoifJtC8q8Rt15dWfN3BuC8dgA4OL
v1q4+GvcWb/e4gsaUj3H3ecdv2m8XhFQWxdJolJHUl0f0WzFqxiAoduJnXcc
sik5SfDIE+trrlsp0ip+sVUntOI+Ro9o5/AXjeT3t6/atGwlEY4qOa/6yqte
zN1ts9lxbuvH4Hm9m9fbxO86lTpaPamUrS+ol3y6Iur/yq8ko+ovov4mqNmV
TeUL8pEn1g+4mM36W28arnSVVNykuIV115S2ojdFPaNGtNetd3g6gnvOO54L
3zaB0SPaaz6NrT+MGtFes0hw/dPBTtsPYfSIdicUsyqNfnyGk0oLO/HA0Zz7
5pdu+mIfPaK95i+OWsmnK5XEVC8ZVaafeOBo/nTWEdx93vGbelEuq9GJqBJz
VxXzrpw3q62Rnw1X1Le4MnffV3f90lWFe/X9JmV7Re6u/qVYRl8sx+2umXIZ
xZgrLcyqVRLh1u7LbFvUyIYrTiqDRDH5lL0hsCf9kFUvW105WG+9WsmoUu8x
okbFb1GtmLvrkLNZDQ/KxtyV/q772RaOwXqu0Y/PcEV9C1fUb42B1PSz7D00
ZRNVT+7JaUYLmq5i7mqbzY5zWz8Gz3Prr343mJOKmVlvufWXmZn1iaYmFUnH
SlosaYmks+osM1HSrZIWSJpZmH6PpHl5Xmdh+jmSVuTpt0p6Q2He2XlfiyUd
08xjMzOzLTWtol5SG/Bd4ChgOXCLpKkRsbCwzAjgIuDYiFgmafeqzRweEatr
bP7CiPhq1f72A04C9gdGAb+VtE9E9N3t4GZmg1wzr1QOBpZExNKIeAq4Gjih
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
" /></p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Computing the performance of the modele...</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">Training accuracy: 99.79,        Validation accuracy: 98.98,        Test accuracy: 98.69</p>
<p style="-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><br /></p>
</div>
