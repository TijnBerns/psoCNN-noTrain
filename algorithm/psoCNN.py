import numpy as np
from copy import deepcopy
import torchvision
from population import Population
from torch.utils.data import DataLoader

class psoCNN:
    def __init__(self, config):
        self.device = config.device

        self.pop_size = config.population_size
        self.n_iter = config.number_iterations
        self.epochs = config.epochs_pso

        self.batch_size = config.batch_size_pso
        self.gBest_acc = np.zeros(self.n_iter)
        self.gBest_measure = np.zeros(self.n_iter)
        self.gBest_test_acc = np.zeros(self.n_iter)
        
        self._init_dataset(config)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True) 
        self.test_dl = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False) 

        print("Initializing population...")
        self.population = Population(config)

        print("Verifying accuracy of the current gBest...")
        print(self.population.particle[0])
        self.gBest = deepcopy(self.population.particle[0])
        self.gBest.model_compile(config.dropout)                          
        acc, score = self.gBest.model_fit(self.train_dl, epochs=self.epochs)       
        test_metrics = self.gBest.model_evaluate(self.test_dl)        
        self.gBest.model_delete()

        self.population.particle[0].acc = acc   
        self.population.particle[0].score = score                        
        self.population.particle[0].pBest.acc = acc    
        
        self.gBest_acc[0] = acc      
        self.gBest_measure[0] = self.population.particle[0].measure()                                       
        self.gBest_test_acc[0] = test_metrics[1]                        
        
        print("Current gBest acc: " + str(self.gBest_acc[0]))
        print("Current gBest measure: " + str(self.gBest_measure[0]))
        print("Current gBest test acc: " + str(self.gBest_test_acc[0]) + "\n")

        print("Looking for a new gBest in the population...")
        for i in range(1, self.pop_size):
            print('Initialization - Particle: ' + str(i+1))
            print(self.population.particle[i])

            self.population.particle[i].model_compile(config.dropout)
            acc, score = self.population.particle[i].model_fit(self.train_dl, epochs=self.epochs) # <<<<<
            self.population.particle[i].model_delete()

            self.population.particle[i].acc = acc
            self.population.particle[i].pBest.score = score
            self.population.particle[i].pBest.acc = acc

            if self.population.particle[i].pBest.measure() >= self.gBest_measure[0]:
                print("Found a new gBest.")
                self.gBest = deepcopy(self.population.particle[i])
                self.gBest_acc[0] = self.population.particle[i].pBest.acc
                self.gBest_measure[0] = self.population.particle[i].pBest.measure()
                print("New gBest acc: " + str(self.gBest_acc[0]))
                print("New gBest measure: " + str(self.gBest_measure[0]))

                self.gBest.model_compile(config.dropout)
                test_metrics = self.gBest.model_evaluate(self.test_dl)
                self.gBest_test_acc[0] = test_metrics[1]
                print("New gBest test acc: " + str(self.gBest_acc[0]))

            self.gBest.model_delete()
            
    def _init_dataset(self, config):
        if config.dataset == "mnist":
            self.train_ds = torchvision.datasets.MNIST(
                root=config.data_path, train=True, download=False, transform=config.transform)
            self.test_ds = torchvision.datasets.MNIST(
                root=config.data_path, train=False, download=False, transform=config.transform)
            
        elif config.dataset == "cifar10":
            self.train_ds = torchvision.datasets.CIFAR10(
                root=config.data_path, train=True, download=False, transform=config.transform)
            self.test_ds = torchvision.datasets.CIFAR10(
                root=config.data_path, train=False, download=False, transform=config.transform)
        else: 
            raise NotImplementedError
        

    def fit(self, Cg, dropout_rate):
        for i in range(1, self.n_iter):
            gBest_acc = self.gBest_acc[i-1]
            gBest_measure = self.gBest_measure[i-1]
            gBest_test_acc = self.gBest_test_acc[i-1]

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
                acc, score = self.population.particle[j].model_fit(self.train_dl, self.epochs)
                self.population.particle[j].model_delete()

                self.population.particle[j].acc = acc
                self.population.particle[j].score = score

                # f_test = self.population.particle[j].acc
                # pBest_acc = self.population.particle[j].pBest.acc
                
                f_test = self.population.particle[j].measure()
                pBest_acc = self.population.particle[j].pBest.acc
                pBest_measure = self.population.particle[j].pBest.measure()

                if f_test >= pBest_measure:
                    print("Found a new pBest.")
                    print("Current acc: " + str(f_test))
                    print("Past pBest acc: " + str(pBest_acc))
                    pBest_acc = acc
                    self.population.particle[j].pBest = deepcopy(
                        self.population.particle[j])

                    if pBest_measure >= gBest_measure:
                        print("Found a new gBest.")
                        gBest_acc = pBest_acc
                        gBest_measure = pBest_measure
                        self.gBest = deepcopy(self.population.particle[j])

                        self.gBest.model_compile(dropout_rate)
                        acc, score = self.gBest.model_fit(self.train_dl, epochs=self.epochs)
                        test_metrics = self.gBest.model_evaluate(self.test_dl)
                        self.gBest.model_delete()
                        gBest_test_acc = test_metrics[1]

            self.gBest_acc[i] = gBest_acc
            self.gBest_test_acc[i] = gBest_test_acc

            print("Current gBest acc: " + str(self.gBest_acc[i]))
            print("Current gBest test acc: " + str(self.gBest_test_acc[i]))
            print("Current gBest measure: " + str(self.gBest_measure[i]))


    def fit_gBest(self, epochs, dropout_rate):
        print("\nFurther training gBest model...")
        self.gBest.model_compile(dropout_rate)

        # trainable_count = 0
        # for i in range(len(self.gBest.model.trainable_weights)):
        #     trainable_count += keras.backend.count_params(
        #         self.gBest.model.trainable_weights[i])
        # print("gBest's number of trainable parameters: " + str(trainable_count))
        
        acc, _ = self.gBest.model_fit_complete(self.train_dl, epochs=epochs)

        # return trainable_count
        return acc

    def evaluate_gBest(self):
        print("\nEvaluating gBest model on the test set...")

        metrics = self.gBest.model_evaluate(self.test_dl)

        print("\ngBest model loss in the test set: " +
              str(metrics[0]) + " - Test set accuracy: " + str(metrics[1]))
        return metrics
    
    