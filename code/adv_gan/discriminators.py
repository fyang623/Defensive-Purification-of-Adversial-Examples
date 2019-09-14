import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, dataset_name):
        super(Discriminator, self).__init__()
        self.dataset_name = dataset_name

        if dataset_name in ['mnist', 'fmnist']:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
            self.in1 = nn.InstanceNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
            self.in2 = nn.InstanceNorm2d(32)
            self.fc = nn.Linear(3 * 3 * 32, 1)

        elif dataset_name == 'cifar10':
            self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
            self.in1 = nn.InstanceNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.in2 = nn.InstanceNorm2d(32)
            self.fc = nn.Linear(2 * 2 * 3 * 32, 1)

        else:
            raise Exception("dataset must be one of mnist, fmnist and cifar10")

    def forward(self, x):
        if self.dataset_name in ['mnist', 'fmnist']:
            x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
            x = F.leaky_relu(self.in1(self.conv2(x)), negative_slope=0.2)
            x = F.leaky_relu(self.in2(self.conv3(x)), negative_slope=0.2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
            x = F.leaky_relu(self.in1(self.conv2(x)), negative_slope=0.2)
            x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
            x = F.leaky_relu(self.in2(self.conv4(x)), negative_slope=0.2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x
