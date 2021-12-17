import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os

from torch import nn, sigmoid, relu, tanh, softmax, log_softmax, optim
from torch.utils.data import TensorDataset
from mnist import MNIST


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
acts_list = {'sigmoid': sigmoid, 'relu': relu, 'tanh': tanh, 'softmax': softmax, 'log_softmax': log_softmax}

def save2disk(arrays, filenames):
    """
    Save arrays to disk
    """
    for i, array in enumerate(arrays):
        np.save(filenames[i], array)

def load_from_disk(*files, dirname='data'):
    """
    Load numpy arrays from files
    """
    np_arrays = []
    for file in files:
        np_arrays.append(np.load(dirname + '/' + str(file) + '.npy'))
    return set_dtype(np.float32, *np_arrays)

def load_mnist_data():
    """
    Load raw data from local machine mnist file
    """
    mndata = MNIST('data')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    # Convert data to numpy.ndarray objects
    all_images = np.asarray(train_images + test_images, dtype=np.float32).reshape(-1, 784) / 255
    all_labels = np.asarray(train_labels + test_labels, dtype=np.int64).reshape(-1)

    # Shuffle all data
    p = np.random.permutation(all_labels.shape[0])
    return all_images[p], nmbr2vec(all_labels[p], 10)


def nmbr2vec(classes, n_classes=None):
    """
    Convert numpy array of number labels to numpy array of vector labels
    """
    classes_vec = np.zeros((classes.shape[0], n_classes))
    for i in range(classes.shape[0]):
        classes_vec[i, classes[i]] = 1
    return classes_vec

def set_dtype(dtype, *np_arrays):
    """
    Change collectivity the dtype of numpy arrays
    """
    arrays = []
    for arr in np_arrays:
        arrays.append(np.asarray(arr, dtype=dtype))
    return arrays

class FCNet(nn.Module):
    """
    Full Connected Neural network
    """
    def __init__(self, fc_layout=None, init=None):
        super().__init__()
        self.layers = []
        self.layers = nn.ModuleList(self.layers)
        for i in range(len(fc_layout) - 1):
            self.layers.append(nn.Linear(fc_layout[i], fc_layout[i+1]))
            if init == 'he':
                torch.nn.init.xavier_normal_(self.layers[i].weight)

    def forward(self, x):
        """
        Compute the outputs of layers
        """
        for i in range(len(self.layers) - 1):
            x = relu(self.layers[i](x))
        return softmax(self.layers[-1](x), dim=1)


class ConvNet(nn.Module):
    """
    Convolutional neural network
    """
    def __init__(self, cv_maxpool=None, cv_layout=None, cv_kernels=None, cv_strides=None, cv_padding=None, \
            cv2linear_size=None, out_classes=None):
        """
        Initialise a convolutional neural network
        """
        super().__init__()
        self.cv_layers = []
        self.cv_layers = nn.ModuleList(self.cv_layers)
        for i in range(len(cv_layout) - 1):
            self.cv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=cv_layout[i], out_channels=cv_layout[i+1], kernel_size=cv_kernels[i], stride=cv_padding[i], padding=cv_padding[i]),
                nn.ReLU(inplace=True)))
            if cv_maxpool[i]:
                self.cv_layers[-1].add_module("maxpool", nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
        self.cv2linear = nn.AdaptiveAvgPool2d(cv2linear_size)
        self.linear = nn.Linear(cv2linear_size[0]*cv2linear_size[1]*cv_layout[-1], out_classes)
        for m in self.modules():
            try:
                for l in m:
                    if isinstance(l, torch.nn.Conv2d):
                        nn.init.kaiming_normal_(l.weight.detach())
                        l.bias.detach().zero_()
                    elif isinstance(l, torch.nn.Linear):
                        nn.init.kaiming_normal_(l.weight.detach())
                        l.bias.detach().zero_()
            except Exception:
                if isinstance(m, torch.nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.detach())
                    m.bias.detach().zero_()
                elif isinstance(m, torch.nn.Linear):
                    nn.init.kaiming_normal_(m.weight.detach())
                    m.bias.detach().zero_()

    def forward(self, x):
        """
        Compute the outputs of layers
        """
        for i in range(len(self.cv_layers)):
            x = self.cv_layers[i](x)
        return softmax(self.linear(torch.flatten(self.cv2linear(x), start_dim=1, end_dim=-1)), dim=1)


def fc(net, dataset, digit=2):
    """
    Round the accuracy of a network model for a dataset
    """
    return round(compute_accuracy(net, dataset)*100, digit)

def compute_accuracy(net, dataset):
    """
    compute the accuracy of a model for a dataset
    """
    with torch.no_grad():
        total, correct = 0, 0
        for batch in dataset:
            X, y = batch
            ypred = net(X)
            correct += torch.sum(torch.argmax(ypred, dim=1) == torch.argmax(y, dim=1))
            total += y.shape[0]
    return float(correct / total)

def seeds_rngs(seed=648712694):
    """
    Initialise all used rng for result repeatability
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_deterministic():
    """
    Set the behavior of torch to deterministic
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

def train_net(net=None, trainset=None, validset=None, loss_func=None, optimizer=None, n_epochs=10, batch_log=32, learn_decay=0.94, debug=True, debug_epoch=1):
    """
    Train neural network
    """
    def fq(u,d):
        return f'{u:{len(str(d))}d}/{d:{len(str(d))}d}'
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learn_decay)
    n_batchs = len(trainset)
    loss_history, i = torch.zeros(n_epochs, 1), 0
    for epoch_idx in range(n_epochs):
        print(f'Epoch {epoch_idx + 1} -------------------------------------------------------------')
        for batch_idx, batch in enumerate(trainset):
            X, y = batch
            for param in net.parameters():
                param.grad = None
            ypred = net(X)
            loss = loss_func(ypred, y)
            loss.backward()
            optimizer.step()
            if debug and ((batch_idx+1) & (batch_log-1)) == 0:
                print (f'Epoch: {fq(epoch_idx+1, n_epochs)} | Batch: {fq(batch_idx+1, n_batchs)} | Cost: {float(loss):.6f}')
        scheduler.step()
        loss_history[i], i = float(loss), i + 1
        if (epoch_idx+1) & (debug_epoch-1) == 0:
            print(f'>> Cost: {float(loss):.6f}, Training accuracy: {fc(net, trainset)}, Validation accuracy: {fc(net, validset)}\n')
    return loss_history



def split2tvt(all_X, all_y, train_ratio=0.8, valid_ratio=0.2, dataset=None):
    """
    Split dataset to train, valid and test
    """
    total_samples = all_X.shape[0]
    ranges = (int(total_samples * train_ratio * (1-valid_ratio)), int(total_samples * train_ratio))
    all_X = torch.from_numpy(all_X).to(device)
    all_y = torch.from_numpy(all_y).to(device)
    return TensorDataset(all_X[:ranges[0], :], all_y[:ranges[0]]), \
            TensorDataset(all_X[ranges[0]:ranges[1], :], all_y[ranges[0]:ranges[1]]), \
            TensorDataset(all_X[ranges[1]:, :], all_y[ranges[1]:])


def check_files(*files):
    for file in files:
        if not os.path.exists(file):
            return False
    return True



def display_samples(X, shape=None):
    if isinstance(X, torch.Tensor):
        X = torch.detach().numpy()
    pic_w, pic_h = X.shape[2], X.shape[3]
    image = np.zeros((shape[0]*pic_w, shape[1]*pic_h))
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i*pic_h:(i+1)*pic_h, j*pic_w:(j+1)*pic_w] = X[i*shape[0]+j].reshape(pic_h, pic_w)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
