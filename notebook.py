#!/usr/bin/python3

# %% Imports

import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import *
from torch.utils.data import DataLoader, TensorDataset



# %% Set general configurations about the number of epochs, learning rate,
# %% neural network type (Convolutional / full-connected), and visualisation
# %% Also, we need to seed the random number generators

print('General setup ...')

n_epochs = 80
lr = 0.001
nn_type = "cv"
ld =  0.97
visualize_data = True
forceReloadData = False # force Reloading of all the data from raw files
seeds_rngs()

# %% Configure the training of the GPU:0, if not available do it on the CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% Start loading and preparing the train set, the validation set and the
# %% test set, the data is all ready prepared in the all_images.npy and 
# %% all_labels.npy in the folder `data`.
# %% if there is any need for loading the data from the raw file, we use
# %% the wrapper function load_mnist_data to do that

print('Load and prepare data ...')
filenames = ['all_images', 'all_labels'] # no need to add '.npy' extension
if check_files(*filenames) and not forceReloadData: # we check if files exist
    all_X, all_y = load_from_disk(*filenames)
else:
    all_X, all_y = load_mnist_data()


# %% Reshape the data if we using a Convnet from (-1, 28*28) to (-1, 1, 28, 28)
if nn_type == "cv":
    all_X = all_X.reshape(-1, 1, 28, 28)

# %% Visualise some of the dataset
if visualize_data:
    display_samples(all_X[:20*30], shape=(20, 30))


# %% Splitting the data to trainset, validset, testset
train_X_y, valid_X_y, test_X_y = split2tvt(all_X, all_y)
# We don't need any more those arrays
del all_X, all_y

# %% Construting datasets wrappers
print('Construting datasets ...')
trainset = DataLoader(train_X_y, batch_size=64, shuffle=False)
validset = DataLoader(valid_X_y, batch_size=64, shuffle=False)
testset = DataLoader(test_X_y, batch_size=64, shuffle=False)
# We don't need any more those arrays
del train_X_y, valid_X_y, test_X_y

# %% Defining  the model of the neural network
print('Defining neural networks ...')

if nn_type == "fc":
    # if we want to use a fully connected neural network
    fc_layout = (784, 64, 64, 10) # (input, fist hidden, second hidden, output)
    # we pass the layout to the FCNet function that do the magic
    # we use he initialisation as a initialisation strategy
    # the training is being done in the device
    net = FCNet(fc_layout, init="he").to(device)
elif nn_type == "cv":
    # if we want to use a convolutionnal neural network
    cv_kernels = ((4, 4), (4, 4), (2, 2), (2, 2), (2, 2))
    cv_strides = ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1))
    cv_padding = ((2, 2), (1, 1), (1, 1), (1, 1), (1, 1))
    cv_layout = (1, 8, 16, 32, 64, 128) # 28, 12, 6, 3
    cv_maxpool = [True, False, False, True, True]
    cv2linear_size = (1, 1)
    out_classes = 10
    net = ConvNet(cv_maxpool=cv_maxpool, cv_layout=cv_layout, \
            cv_kernels=cv_kernels, cv_strides=cv_strides, \
            cv_padding=cv_padding, cv2linear_size=cv2linear_size, \
            out_classes=out_classes).to(device)

# %% Setting of the optimizer and cost/loss function
print('Defining optimizer and cost function ...')
# multi-label one-versus-all loss based on max-entropy
loss_func = nn.MultiLabelSoftMarginLoss()
# adam optimizer
optimizer = optim.Adam(params=net.parameters(), lr=lr, betas=(0.9, 0.999), \
        eps=1e-8, weight_decay=0, amsgrad=False)

# %% Start the training of the model
print('Starting training ...')
start = time.time()
loss_history = train_net(net, trainset, validset, loss_func=loss_func,
        optimizer=optimizer, n_epochs=n_epochs, batch_log=128,\
                learn_decay=ld, debug=True, debug_epoch=8)
end = time.time()
print('Finishing training ...')
print(f'Training time: {end-start}')

# %% Plot the loss history of the model
loss_history = loss_history.detach().numpy()
f, a = plt.subplots()
a.scatter(np.arange(max(loss_history.shape)), loss_history)
a.set_title('the cost value of the model during the training')
a.set_xlabel('the training time in term of epoch numbers of training')
a.set_ylabel('the value of the softmax error')
f.canvas.draw()
f.canvas.flush_events()
plt.show()


# %% Evaluating the model on the dataset, validset, testset


train_loss = fc(net, trainset)
valid_loss = fc(net, validset)
test_loss = fc(net, testset)

print(f'\n\nComputing the performance of the modele...')
print(f'\nTraining accuracy: {train_loss},\
        Validation accuracy: {valid_loss},\
        Test accuracy: {test_loss}\n')

# %% Saving model
total_loss_str = str(train_loss) + '_' + str(valid_loss) + '_' + str(test_loss)
if nn_type == 'cv':
    filename = 'models/net_cv_' + total_loss_str + str(time.time())
elif nn_type == 'fc':
    filename = 'models/net_fc_' + total_loss_str + str(time.time())
torch.save(net.state_dict(), filename)
