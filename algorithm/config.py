from re import L
import torch
from torchvision import transforms

class Config():
    def __init__(self) -> None:
        self.version = 2
        self.dataset: str = "cifar10"
        # self.dataset: str = "mnist"
        self.seed = 0
        self.results_path: str = f"../results/{self.dataset}_{self.version}/"
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.device = torch.device("cpu")

        ######## Algorithm parameters ##################
        # self.type = "train"
        self.particle_type = "mellor"
        # self.type = "ntk"
        
        self.number_runs: int = 1
        self.number_iterations: int = 20
        self.population_size: int = 10

        self.batch_size_pso: int = 32
        self.batch_size_full_training: int = 32

        self.epochs_pso: int = 1
        self.epochs_full_training: int = 50

        self.max_conv_output_channels: int = 256
        self.max_fully_connected_neurons: int = 300

        self.min_layer: int = 3
        self.max_layer: int = 20

        self.probability_convolution: float = 0.6
        self.probability_pooling: float = 0.3
        self.probability_fully_connected: float = 0.1

        self.max_conv_kernel_size: int = 7

        self.Cg: float = 0.5
        self.dropout: float = 0.5
        ######## Dataset parameters ##################

        self.data_path: str = "./data/data"

        if self.dataset == "mnist":
            self.input_width = 28
            self.input_height = 28
            self.input_channels = 1
            self.output_dim = 10
            self.transform = transforms.ToTensor()
        
        elif self.dataset == "cifar10":
            self.input_width = 32
            self.input_height = 32
            self.input_channels = 3
            self.output_dim = 10
            self.transform = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                                 transforms.RandomHorizontalFlip(), 
                                                 transforms.ToTensor()],)
