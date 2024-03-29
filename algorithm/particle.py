from cmath import inf
from re import A
import numpy as np
import utils
from copy import deepcopy
from tqdm import tqdm
import torch.optim as optim

import torch
import torch.nn as nn

from ntk import NTK
import mellor


class Particle():
    def __init__(self, config, max_pool_layers):

        self.min_layer = config.min_layer
        self.max_layer = config.max_layer

        self.num_pool_layers = 0
        self.max_pool_layers = max_pool_layers

        self.input_width = config.input_width
        self.input_height = config.input_height
        self.input_channels = config.input_channels

        self.depth = np.random.randint(config.min_layer, config.max_layer)
        self.conv_prob = config.probability_convolution
        self.pool_prob = config.probability_pooling
        self.fc_prob = config.probability_fully_connected

        self.max_conv_kernel = config.max_conv_kernel_size
        self.max_out_ch = config.max_conv_output_channels
        self.max_fc_neurons = config.max_fully_connected_neurons

        self.output_dim = config.output_dim

        self.particle_type = config.particle_type
        self.train = config.train

        self.layers = []
        self.acc = None
        self.score = None
        self.vel = []
        self.pBest = []

        self.device = config.device

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
        self.layers.append(
            {"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})

        conv_prob = self.conv_prob
        pool_prob = conv_prob + self.pool_prob
        fc_prob = pool_prob

        for _ in range(1, self.depth):
            if self.layers[-1]["type"] == "fc":
                layer_type = 1.1
            else:
                layer_type = np.random.rand()

            if layer_type < conv_prob:
                self.layers = utils.add_conv(
                    self.layers, self.max_out_ch, self.max_conv_kernel)

            elif layer_type >= conv_prob and layer_type <= pool_prob:
                self.layers, self.num_pool_layers = utils.add_pool(
                    self.layers, self.num_pool_layers, self.max_pool_layers)

            elif layer_type >= fc_prob:
                self.layers = utils.add_fc(self.layers, self.max_fc_neurons)

        self.layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}

    def velocity(self, gBest, cg):
        self.vel = utils.computeVelocity(
            gBest, self.pBest.layers, self.layers, cg)

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
                    updated_list_layers.append(
                        {"type": "conv", "ou_c": list_layers[i]["ou_c"], "kernel": list_layers[i]["kernel"]})

                if list_layers[i]["type"] == "fc":
                    updated_list_layers.append(list_layers[i])

                if list_layers[i]["type"] == "max_pool":
                    updated_list_layers.append(
                        {"type": "max_pool", "ou_c": -1, "kernel": 2})

                if list_layers[i]["type"] == "avg_pool":
                    updated_list_layers.append(
                        {"type": "avg_pool", "ou_c": -1, "kernel": 2})

        return updated_list_layers

    def model_compile(self, dropout_rate):
        list_layers = []

        for i in range(len(self.layers)):
            if self.layers[i]["type"] == "conv":
                out_c = self.layers[i]["ou_c"]
                kernel_size = self.layers[i]["kernel"]

                if i == 0:
                    in_c = self.input_channels
                    list_layers.append(
                        nn.Conv2d(in_c, out_c, kernel_size, stride=1, padding=5))
                else:
                    list_layers.append(nn.LazyConv2d(
                        out_c, kernel_size, stride=1, padding=5))

                list_layers.append(nn.LazyBatchNorm2d())
                list_layers.append(nn.ReLU())

            if self.layers[i]["type"] == "max_pool":
                kernel_size = self.layers[i]["kernel"]
                list_layers.append(nn.MaxPool2d(
                    kernel_size=kernel_size, stride=2))

            if self.layers[i]["type"] == "avg_pool":
                kernel_size = self.layers[i]["kernel"]
                list_layers.append(nn.AvgPool2d(
                    kernel_size=kernel_size, stride=2))

            if self.layers[i]["type"] == "fc":
                if self.layers[i-1]["type"] != "fc":
                    list_layers.append(nn.Flatten())

                out_features = self.layers[i]["ou_c"]

                list_layers.append(nn.Dropout(dropout_rate))
                list_layers.append(nn.LazyLinear(out_features))
                list_layers.append(nn.LazyBatchNorm1d())

                if i == len(self.layers[i]) - 1:
                    list_layers.append(nn.Softmax(dim=0))
                else:
                    list_layers.append(nn.ReLU())

        self.model = nn.Sequential(*list_layers)

    def _epoch(self, loader, opt=None):
        total_acc = 0
        total_loss = 0
        ce = nn.CrossEntropyLoss()
        loss = -1
        desc = "validating particle" if opt is None else "training particle"
        pbar = tqdm(loader, desc=desc)
        for (x, y) in pbar:
            # Prediction
            x, y = x.to(self.device), y.to(self.device)
            yp = self.model(x)
            loss = ce(yp, y)

            # Backpropagate loss
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

            # Update error and loss
            total_acc += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * x.shape[0]

            # pbar.set_description(desc + f" loss: {loss:.2f}")

        return total_loss / len(loader.dataset), total_acc / len(loader.dataset)
    
    def measure(self):
        if self.particle_type == "regular":
            return self.acc
        else:
            return self.score
    

    def model_fit_train(self, loader, epochs):
               
        loss = 0
        best_loss = 1e12
        acc = 0
        best_params = None
        self.model.train()
        self.model = self.model.to(self.device)

        adam_opt = optim.Adam(self.model.parameters(),
                              lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)

        for _ in tqdm(range(epochs), desc=f"Training model for {epochs} epochs"):
            loss, acc = self._epoch(loader=loader, opt=adam_opt)
            if loss < best_loss:
                best_params = self.model.state_dict()

        self.model.load_state_dict(best_params)
        self.model = self.model.to("cpu")
        self.model
        return loss, acc

    def model_ntk(self, loader):
        ntk = NTK(self.device).get_ntk_score(loader, self.model, 1)
        score = 1 / ntk
        return score
    
    def model_mellor(self, loader):
        score = mellor.score_network(self.model, loader)
        return score

    def model_fit(self, loader, epochs):
        if self.particle_type == "ntk":
            score = self.model_ntk(loader) 
        elif self.particle_type == "mellor":
            score = self.model_mellor(loader) 
        else:
            score = -inf
            
        self.model_delete()
        torch.cuda.empty_cache()
        self.model_compile(0.5)
        
        if self.train:
            _, acc = self.model_fit_train(loader, epochs)
        else:
            acc = 0
        
        return acc, score 

    def model_evaluate(self, loader):
        self.model.eval()
        self.model.to(self.device)
        loss, acc = self._epoch(loader)
        return loss, acc

    def model_fit_complete(self, loader, epochs):
        return self.model_fit_train(loader=loader, epochs=epochs)

    def model_delete(self):
        del self.model
        self.model = None
