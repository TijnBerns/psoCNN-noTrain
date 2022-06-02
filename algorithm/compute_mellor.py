import torch.nn as nn
from torch.utils.data import DataLoader
import mellor
from config import Config 
import torchvision

model = nn.Sequential(
    nn.Conv2d(3, 89, kernel_size=(4, 4), stride=(1, 1), padding=(5, 5)),
    nn.BatchNorm2d(89, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(89, 203, kernel_size=(6, 6), stride=(1, 1), padding=(5, 5)),
    nn.BatchNorm2d(203, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Conv2d(203, 254, kernel_size=(3, 3), stride=(1, 1), padding=(5, 5)),
    nn.BatchNorm2d(254, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=260096, out_features=10, bias=True),
    nn.BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU()
)

config = Config()
test_ds = torchvision.datasets.CIFAR10(root="./data/data", train=False, transform=config.transform)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=4)

print(mellor.score_network(model, test_dl))