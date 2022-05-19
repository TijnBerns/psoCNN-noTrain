# create list of networks
# import dataset

# run get_ntk_n()
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from algorithm.fitness.lib.ntk import get_ntk_n
from algorithm.particle import Particle


def get_networks(n):
    nets = []

    m_pool = 0
    in_w = 28
    while in_w > 4:
        m_pool += 1
        in_w = in_w / 2

    for i in range(n):
        particle = Particle(min_layer=3, max_layer=20, max_pool_layer=m_pool, width_in=28, height_in=28, channels_in=1,
                            conv_prob=0.6, pool_prob=0.3, max_conv_kernels=7, max_ch_out=256, max_fc_neurons=300,
                            dim_out=10)
        particle.model_compile()
        model = particle.model
        nets.append(model)

    return nets


def get_dataloader():
    root = './data'
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = MNIST(root=root, train=True, transform=trans, download=True)
    dataloader = torch.utils.data.DataLoader(train_set)

    return dataloader


networks = get_networks(2)
dataloader = get_dataloader()

get_ntk_n(dataloader, networks)
