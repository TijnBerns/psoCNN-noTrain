# create list of networks
# import dataset

# run get_ntk_n()
import sys
sys.path.append('../algorithm')

import torch
from torch import nn
from torch.utils.data import TensorDataset
import torchvision.datasets as datasets
from torchvision.transforms import transforms
import torch.nn.functional as F

from linear_regions import get_linear_regions
from ntk import NTK 
from particle import Particle

device = "cpu"

class NetMnist(nn.Module):
    def __init__(self):
        super(NetMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class VerySimple(nn.Module):
    def __init__(self) -> None:
        super(VerySimple, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 10)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.fc1(x)


def get_networks(n):
    nets = []

    m_pool = 0
    in_w = 28
    while in_w > 4:
        m_pool += 1
        in_w = in_w / 2

    for i in range(n):
        particle = Particle(min_layer=3, max_layer=20, max_pool_layers=m_pool, input_width=28, input_height=28,
                            input_channels=1, conv_prob=0.6, pool_prob=0.3, fc_prob=0.1, max_conv_kernel=7,
                            max_out_ch=256, max_fc_neurons=300, output_dim=10, device=device)
        particle.model_compile(dropout_rate=0.5)
        model = particle.model
        nets.append(model)

    return nets


def get_dataloader(batch_size=32):
    root = '../data'
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR100(root=root, train=True, transform=trans, download=False)
    #train_set.train_data.to(torch.device("cuda"))
    #train_set.train_labels.to(torch.device("cuda"))
    #tensor_set = TensorDataset(train_set)

    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return dataloader

#networks = get_networks(2)
network = VerySimple()
network.to(device)
networks = [network]
# networks = get_networks(1)
# network.cuda()
print(network)
dataloader = get_dataloader()

#ntk = get_ntk_n(dataloader, networks, num_batch=1)
for _ in range(10):
    dataloader = get_dataloader()
    ntk = NTK(device).get_ntk_score(dataloader, networks[0], 10)
    print(ntk)
#linear_regions = get_linear_regions(dataloader, networks)
#print(linear_regions)