from particle import Particle
import config


class Population:
    def __init__(self, config):
        # Compute maximum number of pooling layers for any given particle
        max_pool_layers = 0
        in_w = config.input_width

        while in_w > 4:
            max_pool_layers += 1
            in_w = in_w/2

        self.particle = []
        for _ in range(config.population_size):
            self.particle.append(Particle(config, max_pool_layers))
