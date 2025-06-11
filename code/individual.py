import numpy as np
import random
from layers import ConvLayer, PoolLayer, FullLayer
from utils import flip, rand

class Individual:
    def __init__(self, x_prob: float = 0.9, x_eta: float = 0.05, m_prob: float = 0.2, m_eta: float = 0.05):
        self.indi = []
        self.x_prob = x_prob
        self.x_eta = x_eta
        self.m_prob = m_prob
        self.m_eta = m_eta
        self.mean = 0
        self.std = 0
        self.complxity = 0

        self.feature_map_size_range = (3, 50)
        self.filter_size_range = (2, 20)
        self.pool_kernel_size_range = (1, 2)
        self.hidden_neurons_range = (1000, 2000)
        self.mean_range = (-1, 1)
        self.std_range = (0, 1)

    def clear_state_info(self):
        self.complxity = 0
        self.mean = 0
        self.std = 0

    def initialize(self):
        self.indi = self._init_individual()

    def _init_individual(self):
        num_conv = np.random.randint(1, 3)
        num_pool = np.random.randint(1, 3)
        num_full = np.random.randint(1, 3)

        layers = [self._random_conv_layer() for _ in range(num_conv)]
        layers += [self._random_pool_layer() for _ in range(num_pool)]
        layers += [self._random_full_layer() for _ in range(num_full - 1)]
        layers.append(self._common_output_layer())
        return layers

    def get_layer_at(self, i):
        return self.indi[i]

    def get_layer_size(self):
        return len(self.indi)

    # Initialization helpers
    def _init_mean(self):
        return np.random.uniform(*self.mean_range)

    def _init_std(self):
        return np.random.uniform(*self.std_range)

    def _init_filter_size(self):
        return np.random.randint(*self.filter_size_range)

    def _init_feature_map_size(self):
        return np.random.randint(*self.feature_map_size_range)

    def _init_kernel_size(self):
        return 2 ** np.random.randint(self.pool_kernel_size_range[0], self.pool_kernel_size_range[1] + 1)

    def _init_hidden_neurons(self):
        return np.random.randint(*self.hidden_neurons_range)

    # Mutation
    def mutation(self):
        if not flip(self.m_prob):
            return

        new_layers = []

        for i in range(self.get_layer_size() - 1):
            current = self.get_layer_at(i)
            if flip(0.5):
                op = self._mutation_operation(rand())
                max_layers = 6
                if op == 0 and len(new_layers) + self.get_layer_size() - i - 1 < max_layers:
                    new_layers.append(self._generate_new_layer(current.type, self.get_layer_size()))
                    new_layers.append(current)
                elif op in [1, 2]:
                    new_layers.append(self._mutate_layer(current))
            else:
                new_layers.append(current)

        if not new_layers:
            new_layers += [self._random_conv_layer(), self._random_pool_layer()]

        new_layers.append(self.get_layer_at(-1))

        if new_layers[0].type != 1:
            new_layers.insert(0, self._random_conv_layer())

        self.indi = new_layers

    def _mutation_operation(self, r: float) -> int:
        if r < 0.33:
            return 1  # modify
        elif r > 0.66:
            return 2  # delete (not implemented)
        return 0  # add

    def _mutate_layer(self, layer):
        if layer.type == 1:
            return self._mutate_conv(layer)
        elif layer.type == 2:
            return self._mutate_pool(layer)
        elif layer.type == 3:
            return self._mutate_full(layer)
        raise ValueError("Unknown layer type")

    # Mutation logic for each layer type
    def _mutate_conv(self, layer, eta):
        new_fms = int(self._pm(*self.filter_size_range, layer.filter_width, eta))
        new_fmn = int(self._pm(*self.feature_map_size_range, layer.feature_map_size, eta))
        new_mean = self._pm(*self.mean_range, layer.weight_matrix_mean, eta)
        new_std = self._pm(*self.std_range, layer.weight_matrix_std, eta)
        return ConvLayer(filter_size=(new_fms, new_fms), feature_map_size=new_fmn, weight_matrix=[new_mean, new_std])

    def _mutate_pool(self, layer, eta):
        ksize_log = np.log2(layer.kernel_width)
        new_ksize = int(2 ** self._pm(*self.pool_kernel_size_range, ksize_log, eta))
        new_pool_type = self._pm(0, 1, layer.kernel_type, eta)
        return PoolLayer(kernel_size=(new_ksize, new_ksize), pool_type=new_pool_type)

    def _mutate_full(self, layer, eta):
        new_hidden = int(self._pm(*self.hidden_neurons_range, layer.hidden_neuron_num, eta))
        new_mean = self._pm(*self.mean_range, layer.weight_matrix_mean, eta)
        new_std = self._pm(*self.std_range, layer.weight_matrix_std, eta)
        return FullLayer(hidden_neuron_num=new_hidden, weight_matrix=[new_mean, new_std])

    def _pm(self, xl, xu, x, eta):
        delta1 = (x - xl) / (xu - xl)
        delta2 = (xu - x) / (xu - xl)
        rand_val = np.random.random()
        mut_pow = 1.0 / (eta + 1.0)
        if rand_val < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand_val + (1.0 - 2.0 * rand_val) * (xy ** (eta + 1))
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand_val) + 2.0 * (rand_val - 0.5) * (xy ** (eta + 1))
            delta_q = 1.0 - val ** mut_pow
        mutated = x + delta_q * (xu - xl)
        return np.clip(mutated, xl, xu)

    # Layer creation methods
    def _common_output_layer(self):
        return FullLayer(hidden_neuron_num=2, weight_matrix=[self._init_mean(), self._init_std()])

    def _random_full_layer(self):
        return FullLayer(hidden_neuron_num=self._init_hidden_neurons(), weight_matrix=[self._init_mean(), self._init_std()])

    def _random_conv_layer(self):
        s = self._init_filter_size()
        return ConvLayer(
            filter_size=(s, s),
            feature_map_size=self._init_feature_map_size(),
            weight_matrix=[self._init_mean(), self._init_std()]
        )

    def _random_pool_layer(self):
        s = self._init_kernel_size()
        return PoolLayer(kernel_size=(s, s), pool_type=np.random.rand())

    def _generate_new_layer(self, current_type, unit_len):
        if current_type == 3 and unit_len == 1:
            return self._random_conv_layer() if random.random() < 0.5 else self._random_pool_layer()
        return self._random_full_layer() if current_type == 3 else (
            self._random_conv_layer() if random.random() < 0.5 else self._random_pool_layer()
        )

    def __str__(self):
        lines = [
            f'Length: {self.get_layer_size()}, Complexity: {self.complxity}',
            f'Mean: {self.mean:.2f}, Std: {self.std:.2f}'
        ]
        for unit in self.indi:
            if unit.type == 1:
                lines.append(f"Conv[{unit.filter_width}x{unit.filter_height}, FM:{unit.feature_map_size}, μ={unit.weight_matrix_mean:.2f}, σ={unit.weight_matrix_std:.2f}]")
            elif unit.type == 2:
                lines.append(f"Pool[{unit.kernel_width}x{unit.kernel_height}, Type:{unit.kernel_type:.2f}]")
            elif unit.type == 3:
                lines.append(f"Full[{unit.hidden_neuron_num}, μ={unit.weight_matrix_mean:.2f}, σ={unit.weight_matrix_std:.2f}]")
        return '\n'.join(lines)
