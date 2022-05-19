import numpy as np
import utils
from copy import deepcopy
from tqdm import tqdm
import torch.optim as optim

import torch
import torch.nn as nn

class Particle():
    def __init__(self, min_layer, max_layer, max_pool_layers, input_width, input_height, input_channels, \
        conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim, device):
       
        self.min_layer = min_layer 
        self.max_layer = max_layer
        
        self.num_pool_layers = 0
        self.max_pool_layers = max_pool_layers
        
        self.input_width = input_width 
        self.input_height = input_height 
        self.input_channels = input_channels
        
        self.depth = np.random.randint(min_layer, max_layer)
        self.conv_prob = conv_prob
        self.pool_prob = pool_prob
        self.fc_prob = fc_prob
        
        self.max_conv_kernel = max_conv_kernel
        self.max_out_ch = max_out_ch
        self.max_fc_neurons = max_fc_neurons
        
        self.output_dim = output_dim
        
        
        
        self.layers = []
        self.acc = None
        self.vel = []
        self.pBest = []
        
        self.device = device
     
        # Initialize particle
        self.initialization() 
        
        # Update initial velocity
        for i in range(len(self.layers)):
            if self.layers[i]["type"] != "fc":
                self.vel.append({"type": "keep"})
            else:
                self.vel.append({"type": "keep_fc"})
        
        self.model = None
        self.pBest = deepcopy(self)
    
    def __str__(self):
        string = ""
        for z in range(len(self.layers)):
            string = string + self.layers[z]["type"] + " | "
        
        return string
        
    def initialization(self):
        out_channel = np.random.randint(3, self.max_out_ch)
        conv_kernel = np.random.randint(3, self.max_conv_kernel)
        
        # First layer is always a convolution layer
        self.layers.append({"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})

        conv_prob = self.conv_prob
        pool_prob = conv_prob + self.pool_prob
        fc_prob = pool_prob

        for _ in range(1, self.depth):
            if self.layers[-1]["type"] == "fc":
                layer_type = 1.1
            else:
                layer_type = np.random.rand()

            if layer_type < conv_prob:
                self.layers = utils.add_conv(self.layers, self.max_out_ch, self.max_conv_kernel)

            elif layer_type >= conv_prob and layer_type <= pool_prob:
                self.layers, self.num_pool_layers = utils.add_pool(self.layers, self.num_pool_layers, self.max_pool_layers)
            
            elif layer_type >= fc_prob:
                self.layers = utils.add_fc(self.layers, self.max_fc_neurons)
            
        self.layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}
    
    def velocity(self, cg):
        self.vel = utils.computeVelocity(self.gBest, self.pBest.layers, self.layers, cg)
    
    def update(self):
        new_p = utils.updateParticle(self.layers, self.vel)
        new_p = self.validate(new_p)
        
        self.layers = new_p
        self.model = None
    
    def validate(self, list_layers):
        # Last layer should always be a fc with number of neurons equal to the number of outputs
        list_layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}

        # Remove excess of Pooling layers
        self.num_pool_layers = 0
        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "max_pool" or list_layers[i]["type"] == "avg_pool":
                self.num_pool_layers += 1
            
                if self.num_pool_layers >= self.max_pool_layers:
                    list_layers[i]["type"] = "remove"


        # Now, fix the inputs of each conv and pool layers
        updated_list_layers = []
        
        for i in range(0, len(list_layers)):
            if list_layers[i]["type"] != "remove":
                if list_layers[i]["type"] == "conv":
                    updated_list_layers.append({"type": "conv", "ou_c": list_layers[i]["ou_c"], "kernel": list_layers[i]["kernel"]})
                
                if list_layers[i]["type"] == "fc":
                    updated_list_layers.append(list_layers[i])

                if list_layers[i]["type"] == "max_pool":
                    updated_list_layers.append({"type": "max_pool", "ou_c": -1, "kernel": 2})

                if list_layers[i]["type"] == "avg_pool":
                    updated_list_layers.append({"type": "avg_pool", "ou_c": -1, "kernel": 2})

        return 
    
    def model_compile(self, dropout_rate):
        list_layers = []
        
        for i in range(len(self.layers)):
            if self.layers[i]["type"] == "conv":
                out_c = self.layers[i]["ou_c"]
                kernel_size = self.layers[i]["kernel"]
                
                if i == 0:
                    # in_w = self.input_width # Not needed ???
                    # in_h = self.input_height # Not needed ???
                    in_c = self.input_channels
                    list_layers.append(nn.Conv2d(in_c, out_c, kernel_size, stride=1, padding=5))
                else:
                    list_layers.append(nn.LazyConv2d(out_c, kernel_size, stride=1, padding=5))
                                
                list_layers.append(nn.LazyBatchNorm2d())
                list_layers.append(nn.ReLU())

            if self.layers[i]["type"] == "max_pool":
                kernel_size = self.layers[i]["kernel"]
                # TODO: Check kernel size, in their implementations (3,3) is always used
                list_layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=2))

            if self.layers[i]["type"] == "avg_pool":
                kernel_size = self.layers[i]["kernel"]
                # TODO: Check kernel size, in their implementations (3,3) is always used
                list_layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=2))
            
            if self.layers[i]["type"] == "fc":
                if self.layers[i-1]["type"] != "fc":
                    list_layers.append(nn.Flatten())
                    
                out_features = self.layers[i]["ou_c"]

                list_layers.append(nn.Dropout(dropout_rate))
                list_layers.append(nn.LazyLinear(out_features))
                list_layers.append(nn.LazyBatchNorm1d())

                if i == len(self.layers[i]) - 1:
                    # list_layers.append(nn.Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                    list_layers.append(nn.Softmax(dim=0))
                else:
                    # self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01), activation=None))
                    list_layers.append(nn.ReLU())
            
        self.model = nn.Sequential(*list_layers)
    
    def _epoch(self, loader, opt=None):
        total_error = 0
        total_loss = 0
        loss = -1
        desc = "validating particle" if opt is None else "training particle"
        # pbar = tqdm(loader, desc=desc) 
        for (x, y) in loader:
            # Prediction
            x, y = x.to(self.device), y.to(self.device)
            yp = self.model(x)
            loss = nn.CrossEntropyLoss()(yp, y)
             
            # Backpropagate loss
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
            
            # Update error and loss
            total_error += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * x.shape[0]
            
            # pbar.set_description(desc + f" loss: {loss:.2f}")
            
        return total_loss / len(loader.dataset), total_error / len(loader.dataset) 
    
    def model_fit(self, loader, epochs):
        loss = 0
        error = 0
        self.model.train()
        self.model = self.model.to(self.device)
        
        adam_opt = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)

        for _ in range(epochs):
            loss, error = self._epoch(loader=loader, opt=adam_opt)
           
        if self.device != "cpu":
            self.model = self.model.to("cpu")
        
        return loss, 1 - error 
    
    def model_evaluate(self, loader):
        self.model.eval()
        self.model.to(self.device)
        loss, acc = self._epoch(loader)
        return loss, acc 
        
    def model_fit_complete(self, loader, epochs):
        # TODO: Check why a seperate method is used for this
        return self.model_fit(loader=loader, epochs=epochs)
    
    def model_delete(self):
        # This is used to free up memory during PSO training
        self.model = self.model.to("cpu")
        del self.model
        self.model = None
        
        