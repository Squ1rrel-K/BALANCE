from torch import nn


class Model_CIFAR10(nn.Module):
    def __init__(self, bias=True):
        super(Model_CIFAR10, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, 5, 1, 1, bias=bias),
                                   nn.ReLU(),
                                   nn.Conv2d(6, 16, 5, 1, 1, bias=bias),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.dense = nn.Sequential(nn.Linear(3136, 1024, bias=bias),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1024, 10, bias=bias))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 3136)
        x = self.dense(x)
        return x

class Model_CIFAR100(nn.Module):
    def __init__(self, bias=True):
        super(Model_CIFAR100, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, 5, 1, 1, bias=bias),
                                   nn.ReLU(),
                                   nn.Conv2d(6, 16, 5, 1, 1, bias=bias),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.dense = nn.Sequential(nn.Linear(3136, 1024, bias=bias),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1024, 100, bias=bias))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 3136)
        x = self.dense(x)
        return x

class Model_MNIST(nn.Module):
    def __init__(self, bias=True):
        super(Model_MNIST, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 5, 1, 1, bias=bias),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 32, 5, 1, 1, bias=bias),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.dense = nn.Sequential(nn.Linear(12 * 12 * 32, 1024, bias=bias),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1024, 10, bias=bias))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 12 * 12 * 32)
        x = self.dense(x)
        return x

class Model_Regression(nn.Module):
    def __init__(self, in_dim=8, out_dim=1, bias=True):
        super(Model_Regression, self).__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, 64, bias=bias),
                                   nn.ReLU(),
                                   nn.Linear(64, 64, bias=bias),
                                   nn.ReLU())
        self.output = nn.Sequential(nn.Linear(64, 64, bias=bias),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(64, out_dim, bias=bias))

    def forward(self, x):
        x = self.input(x)
        x = self.output(x)
        return x