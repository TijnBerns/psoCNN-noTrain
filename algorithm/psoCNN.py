import numpy as np
from copy import deepcopy
import torchvision
import argparse
from population import Population
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

class psoCNN:
    def __init__(self, dataset, n_iter, pop_size, batch_size, epochs, min_layer, max_layer,
                 conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, dropout_rate, root, device):
        self.device = device

        self.pop_size = pop_size
        self.n_iter = n_iter
        self.epochs = epochs

        self.batch_size = batch_size
        self.gBest_acc = np.zeros(n_iter)
        self.gBest_test_acc = np.zeros(n_iter)

        if dataset == "mnist":
            input_width = 28
            input_height = 28
            input_channels = 1
            output_dim = 10

            self.train_ds = torchvision.datasets.MNIST(
                root=root, train=True, download=False, transform=transforms.ToTensor())
            self.test_ds = torchvision.datasets.MNIST(
                root=root, train=False, download=False, transform=transforms.ToTensor())
            
        elif dataset == "cifar10":
            input_width = 32
            input_height = 32
            input_channels = 3
            output_dim = 10

            self.train_ds = torchvision.datasets.CIFAR10(
                root=root, train=True, download=False, transform=transforms.ToTensor())
            self.test_ds = torchvision.datasets.CIFAR10(
                root=root, train=False, download=False, transform=transforms.ToTensor())
        
        else: 
            raise NotImplementedError
        
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True) 
        self.test_dl = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)  

        print("Initializing population...")
        self.population = Population(pop_size, min_layer, max_layer, input_width, input_height, input_channels,
                                     conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim, self.device)

        print("Verifying accuracy of the current gBest...")
        print(self.population.particle[0])
        self.gBest = deepcopy(self.population.particle[0])
        self.gBest.model_compile(dropout_rate)                         
        score = self.gBest.compute_inv_ntk(self.train_dl)       
        # test_metrics = self.gBest.model_evaluate(self.test_dl)          
        self.gBest.model_delete()

        self.gBest_acc[0] = score                                         
        # self.gBest_test_acc[0] = test_metrics[1]                        

        self.population.particle[0].acc = score                           
        self.population.particle[0].pBest.acc = score                     

        print("Current gBest acc: " + str(self.gBest_acc[0]) + "\n")
        # print("Current gBest test acc: " + str(self.gBest_test_acc[0]) + "\n")

        print("Looking for a new gBest in the population...")
        for i in range(1, self.pop_size):
            print('Initialization - Particle: ' + str(i+1))
            print(self.population.particle[i])

            self.population.particle[i].model_compile(dropout_rate)
            score = self.population.particle[i].compute_inv_ntk(self.train_dl)   
            # self.population.particle[i].model_delete()

            self.population.particle[i].acc = score
            self.population.particle[i].pBest.acc = score

            if self.population.particle[i].pBest.acc >= self.gBest_acc[0]:
                print("Found a new gBest.")
                self.gBest = deepcopy(self.population.particle[i])
                self.gBest_acc[0] = self.population.particle[i].pBest.acc
                print("New gBest score: " + str(self.gBest_acc[0]))

                self.gBest.model_compile(dropout_rate)
                score = self.gBest.compute_inv_ntk(self.train_dl)   
                self.gBest_test_acc[0] = score
                print("New gBest test score: " + str(self.gBest_acc[0]))

            self.gBest.model_delete()

    def fit(self, Cg, dropout_rate):
        for i in range(1, self.n_iter):
            gBest_acc = self.gBest_acc[i-1]
            # gBest_test_acc = self.gBest_test_acc[i-1]

            for j in range(self.pop_size):
                print('Iteration: ' + str(i) + ' - Particle: ' + str(j+1))

                # Update particle velocity
                self.population.particle[j].velocity(self.gBest.layers, Cg)

                # Update particle architecture
                self.population.particle[j].update()

                print('Particle NEW architecture: ')
                print(self.population.particle[j])

                # Compute the acc in the updated particle
                self.population.particle[j].model_compile(dropout_rate)
                acc = self.population.particle[j].compute_inv_ntk(self.train_dl) 
                self.population.particle[j].model_delete()

                self.population.particle[j].acc = acc

                f_test = self.population.particle[j].acc
                pBest_acc = self.population.particle[j].pBest.acc

                if f_test >= pBest_acc:
                    print("Found a new pBest.")
                    print("Current acc: " + str(f_test))
                    print("Past pBest acc: " + str(pBest_acc))
                    pBest_acc = f_test
                    self.population.particle[j].pBest = deepcopy(
                        self.population.particle[j])

                    if pBest_acc >= gBest_acc:
                        print("Found a new gBest.")
                        gBest_acc = pBest_acc
                        self.gBest = deepcopy(self.population.particle[j])

                        self.gBest.model_compile(dropout_rate)
                        acc = self.gBest.compute_inv_ntk(self.train_dl) 
                        # test_metrics = self.gBest.model_evaluate(self.test_dl)
                        self.gBest.model_delete()
                        # gBest_test_acc = test_metrics[1]

            self.gBest_acc[i] = gBest_acc
            # self.gBest_test_acc[i] = gBest_test_acc

            print("Current gBest score: " + str(self.gBest_acc[i]))
            # print("Current gBest test acc: " + str(self.gBest_test_acc[i]))

    def fit_gBest(self, batch_size, epochs, dropout_rate):
        print("\nFurther training gBest model...")
        self.gBest.model_compile(dropout_rate)

        # trainable_count = 0
        # for i in range(len(self.gBest.model.trainable_weights)):
        #     trainable_count += keras.backend.count_params(
        #         self.gBest.model.trainable_weights[i])
        # print("gBest's number of trainable parameters: " + str(trainable_count))
        
        _, acc = self.gBest.model_fit_complete(self.train_dl, epochs=epochs)

        # return trainable_count
        return acc

    def evaluate_gBest(self, batch_size):
        print("\nEvaluating gBest model on the test set...")

        metrics = self.gBest.model_evaluate(self.test_dl)

        print("\ngBest model loss in the test set: " +
              str(metrics[0]) + " - Test set accuracy: " + str(metrics[1]))
        return metrics
