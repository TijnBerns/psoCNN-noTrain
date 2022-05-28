import random
import numpy as np
import torch
from config import Config
from scipy import stats

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

config = Config()

def get_batch_jacobian(net, x, target):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))

def score_network(network, loader):
    network.K = np.zeros((config.batch_size_pso, config.batch_size_pso))    
    
    for _, module in network.named_modules():
        for _, module in module.named_modules():
            module.visited_backwards = False
    
    def counting_forward_hook(module, inp, out):
        if not module.visited_backwards:
            return
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1.-x) @ (1.-x.t())
        network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
        

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True
        
    hooks = []
    for _, module in network.named_modules():
        for _, module in module.named_modules():
            if 'ReLU' in str(type(module)):
                hooks.append(module.register_forward_hook(counting_forward_hook))
                hooks.append(module.register_backward_hook(counting_backward_hook))
            
    network = network.to(config.device)
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    scores = []
    # for j in range(args.maxofn):

    for _ in range(1):
        data_iterator = iter(loader)
        x, target = next(data_iterator)
        x2 = torch.clone(x)
        x2 = x2.to(config.device)
        x, target = x.to(config.device), target.to(config.device)
        get_batch_jacobian(network, x, target) 
        network(x2)
        
        _, ld = np.linalg.slogdet(network.K)
        scores.append(ld)        
    
    for hook in hooks:
        hook.remove()
        
    return np.mean(scores)
