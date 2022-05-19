import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
from pdb import set_trace as bp
from datasets import CUTOUT, Dataset2Class, ImageNet16
from operator import mul
from functools import reduce


class LinearRegionCount(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_samples):
        self.ActPattern = {}
        self.n_LR = -1
        self.n_samples = n_samples
        self.ptr = 0
        self.activations = None

    @torch.no_grad()
    def update2D(self, activations):
        n_batch = activations.size()[0]
        n_neuron = activations.size()[1]
        self.n_neuron = n_neuron
        if self.activations is None:
            self.activations = torch.zeros(self.n_samples, n_neuron).cuda()
        self.activations[self.ptr:self.ptr+n_batch] = torch.sign(activations)  # after ReLU
        self.ptr += n_batch

    @torch.no_grad()
    def calc_LR(self):
        res = torch.matmul(self.activations.half(), (1-self.activations).T.half()) # each element in res: A * (1 - B)
        res += res.T # make symmetric, each element in res: A * (1 - B) + (1 - A) * B, a non-zero element indicate a pair of two different linear regions
        res = 1 - torch.sign(res) # a non-zero element now indicate two linear regions are identical
        res = res.sum(1) # for each sample's linear region: how many identical regions from other samples
        res = 1. / res.float() # contribution of each redudant (repeated) linear region
        self.n_LR = res.sum().item() # sum of unique regions (by aggregating contribution of all regions)
        del self.activations, res
        self.activations = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update1D(self, activationList):
        code_string = ''
        for key, value in activationList.items():
            n_neuron = value.size()[0]
            for i in range(n_neuron):
                if value[i] > 0:
                    code_string += '1'
                else:
                    code_string += '0'
        if code_string not in self.ActPattern:
            self.ActPattern[code_string] = 1

    def getLinearReginCount(self):
        if self.n_LR == -1:
            self.calc_LR()
        return self.n_LR


class Linear_Region_Collector:
    def __init__(self, train_loader, models, input_size=(64, 3, 32, 32), sample_batch=100, seed=0):
        self.train_loader = train_loader
        self.models = []
        self.input_size = input_size  # BCHW
        self.sample_batch = sample_batch
        self.input_numel = reduce(mul, self.input_size, 1)
        self.interFeature = []
        self.seed = seed
        self.models = models
        for model in self.models:
            self.register_hook(model)
        self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(models))]
        if input_size is not None or sample_batch is not None:
            if input_size is not None:
                self.input_size = input_size  # BCHW
                self.input_numel = reduce(mul, self.input_size, 1)
            if sample_batch is not None:
                self.sample_batch = sample_batch
            if self.train_loader is not None:
                self.loader = iter(self.train_loader)
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()

    def clear(self):
        self.LRCounts = [LinearRegionCount(self.input_size[0]*self.sample_batch) for _ in range(len(self.models))]
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())  # for ReLU

    def forward_batch_sample(self):
        for _ in range(self.sample_batch):
            try:
                inputs, targets = self.loader.next()
            except Exception:
                del self.loader
                self.loader = iter(self.train_loader)
                inputs, targets = self.loader.next()
            for model, LRCount in zip(self.models, self.LRCounts):
                self.forward(model, LRCount, inputs)
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]

    def forward(self, model, LRCount, input_data):
        self.interFeature = []
        with torch.no_grad():
            model.forward(input_data.cuda())
            if len(self.interFeature) == 0: return
            feature_data = torch.cat([f.view(input_data.size(0), -1) for f in self.interFeature], 1)
            LRCount.update2D(feature_data)


def get_linear_regions(dataloader, model):
    lrc_model = Linear_Region_Collector(dataloader, [model])

    return lrc_model.forward_batch_sample()
