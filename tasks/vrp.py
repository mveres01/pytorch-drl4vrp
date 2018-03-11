# code based in part on
# http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
# and from
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py
# and from
# https://github.com/pemami4911/neural-combinatorial-rl-pytorch/blob/master/tsp_task.py

# TODO: Change model.py to use minibatches where the sequences have a different number of elements.
# Could use a mask to zero out all elements we don't want to consider in the loss function
# NOTE: The paper uses a batch size of 128

import requests
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import os
import numpy as np
import zipfile
import itertools
from collections import namedtuple


class VehicleRoutingDataset(Dataset):

    def __init__(self, num_samples, input_size, max_demand=9, seed=1234):
        super(VehicleRoutingDataset, self).__init__()

        self.num_samples = num_samples
        self.input_size = input_size
        self.max_demand = max_demand

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Driver location will be the first node in each
        locations = np.random.uniform(0, 1, (num_samples, 2, input_size + 1))
        self.static = torch.FloatTensor(locations)

        # Vehicle needs a load > 0
        loads = np.random.randint(3, max_demand + 1, (num_samples, 1, 1))
        loads = loads.repeat(input_size + 1, axis=2) / float(max_demand)

        demands = np.random.randint(0, max_demand + 1, (num_samples, 1, input_size + 1))
        demands = demands / float(max_demand)

        # The depot will be used to refill the vehicle with the initial load
        demands[:, 0, 0] = -loads[:, 0, 0]

        dynamic = np.concatenate((loads, demands), axis=1)
        self.dynamic = torch.FloatTensor(dynamic)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0])

    def update_mask(self, mask, dynamic, chosen_idx):

        assert mask.shape[0] == dynamic.shape[0] == chosen_idx.shape[0] == 1

        # Nodes with 0-demand cannot be chosen
        mask = (dynamic[:, 1] > 0).type(torch.FloatTensor)

        if not mask[:, 1:].byte().any():  # no demand, terminate search
            return mask * 0.
        elif dynamic[0, 0, 0].data.numpy() <= 0:  # vehicle has no load -> refill
            mask = mask * 0.
            mask[:, 0] = torch.FloatTensor([1.])

        return mask

    @staticmethod
    def update_dynamic(dynamic, chosen_idx):

        idx = chosen_idx[0]

        # Agent chooses to return to the depot, so we reset the 'vehicle load'
        # for each dynamic element to be the original capacity
        if idx == 0:
            dynamic[0, 0, :] = -dynamic[0, 1, 0].expand_as(dynamic[0, 1, :])
        else:
            load = dynamic[0, 0, idx].data
            demand = dynamic[0, 1, idx].data

            load_change = torch.max(torch.FloatTensor([0]), load - demand)
            load_change = load_change.expand_as(dynamic[0, 0])

            demand_change = torch.max(torch.FloatTensor([0]), demand - load)

            dynamic[0, 0, :] = Variable(load_change)
            dynamic[0, 1, idx] = Variable(demand_change)

        return dynamic

    @staticmethod
    def reward(static, tour_indices, use_cuda=False):
        """
        Function of: tour_length + number_passengers
        """

        tour_len = Variable(torch.FloatTensor([0]))

        prev_val = static[0, :2, 0]
        for idx in tour_indices[0]:

            cur_val = static[0, :2, idx]
            tour_len = tour_len + torch.sqrt(torch.sum(torch.pow(prev_val - cur_val, 2)))
            prev_val = cur_val

        return tour_len.unsqueeze(0)

    @staticmethod
    def render(static, tour_indices):
        import matplotlib.pyplot as plt

        plt.close('all')

        num_plots = min(int(np.sqrt(len(tour_indices))), 3)

        for i in range(num_plots ** 2):

            # Convert the indices back into a tour
            idx = tour_indices[i]
            if len(idx.size()) == 1:
                idx = idx.unsqueeze(0)

            idx = idx.expand(static.size(1), -1)
            data = torch.gather(static[i].data, 1, idx).numpy()

            plt.subplot(num_plots, num_plots, i + 1)

            plt.plot(data[0], data[1])
            plt.scatter(data[0], data[1], s=4, c='r')

        plt.tight_layout()


if __name__ == '__main__':

    import sys
    sys.path.append('..')
    from model import DRL4VRP

    task = 'carpool'
    train_size = 100000
    val_size = 1000
    batch_size = 64
    static_size = 2
    dynamic_size = 2
    hidden_size = 128
    dropout = 0.3
    use_cuda = False
    num_layers = 1
    critic_beta = 0.9
    max_grad_norm = 2.
    actor_lr = 5e-5
    actor_decay_step = 5000
    actor_decay_rate = 0.96
    plot_every = 10
    num_nodes = 10

    model = DRL4VRP(static_size, dynamic_size, hidden_size, dropout,
                    num_layers, critic_beta, max_grad_norm,
                    actor_lr, actor_decay_step, actor_decay_rate, plot_every, use_cuda)

    for epoch in range(100):

        train = VehicleRoutingDataset(100000, num_nodes, max_demand=9)
        valid = VehicleRoutingDataset(1000, num_nodes, max_demand=9)

        reward_fn = train.reward
        #model.train(train, valid, reward_fn, batch_size)
        model.train_multilength(train, valid, reward_fn, batch_size)
