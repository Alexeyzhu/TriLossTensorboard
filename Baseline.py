import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 100)
        self.dense1_bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        """
        Forward path architecture
        :param x: input picture
        :return: probs of results out of 10
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model, device, train_loader, optimizer):
    """
    Train step of the model
    :param model: pytorch model
    :param device: where to test model - on cpu or on cuda
    :param train_loader: dataser loader with train dataset
    :param optimizer: pytorch optimizer
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # optimizer step
        optimizer.zero_grad()
        output = model(data)
        # compute loss
        loss = F.nll_loss(output, target)
        # backward step
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    """
    Test step of model
    :param model: pytorch model
    :param device: where to test model - on cpu or on cuda
    :param test_loader: dataser loader with test dataset
    :return: accuracy on test dataset
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return 100. * correct / len(test_loader.dataset)


def set_seed(seed):
    """
    Enables reproducibility of results of torch project
    :param seed: seed for randomizers
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    # enable reproducible result
    set_seed(20)
    batch_size = 32
    test_batch_size = 100

    # SVHN Transformations
    svhn_transform = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # MNIST Transformations
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Data Source SVHN train
    svnh_train = datasets.SVHN('../data', split='train', download=True,
                               transform=svhn_transform)

    # Data Source MNIST test
    mnist_test = datasets.MNIST('../data', train=False, download=True,
                                transform=mnist_transform)

    # SVHN train set Data loader
    train_loader = DataLoader(svnh_train,
                              batch_size=batch_size, shuffle=True)

    # MNIST test set Data loader
    mnist_test_loader = DataLoader(mnist_test,
                                   batch_size=test_batch_size, shuffle=True)

    svnh_test = datasets.SVHN('../data', split='test', download=True,
                              transform=svhn_transform)

    svhn_test_loader = DataLoader(svnh_test,
                                  batch_size=batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # number of epochs
    epochs = 3
    # interval for logging
    log_interval = 700

    model = Net().to(device)

    optimizer = optim.Adam(model.parameters())

    # total accuracy
    accur = 0

    # train/test pipeline
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        accur = test(model, device, mnist_test_loader)

    print(accur)
