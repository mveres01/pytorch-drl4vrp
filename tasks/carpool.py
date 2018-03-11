# code based in part on
# http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
# and from
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py
# and from
# https://github.com/pemami4911/neural-combinatorial-rl-pytorch/blob/master/tsp_task.py

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


class ConstDemandCarpoolDataset(Dataset):
    # TODO: Break down the start  & stop locations so we can spread information better

    def __init__(self, num_samples, input_size, max_num_seats=5, seed=1234):
        super(ConstDemandCarpoolDataset, self).__init__()

        self.num_samples = num_samples
        self.input_size = input_size
        self.max_num_seats = max_num_seats

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Driver location will be the first node in each
        start_locations = np.random.uniform(0, 1, (num_samples, 2, input_size + 1))

        #end_locations = np.random.uniform(0, 1, (num_samples, 2, input_size + 1))

        end_locations = np.random.uniform(0, 1, (num_samples, 2, 1))
        end_locations = end_locations.repeat(input_size + 1, axis=2)

        locations = np.concatenate((start_locations, end_locations), axis=1)
        self.static = torch.FloatTensor(locations)

        is_active = np.zeros((num_samples, 1, input_size + 1))
        is_active[:, 0, 0] = 1.
        self.dynamic = torch.FloatTensor(is_active)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx],
                self.dynamic[idx],
                self.static[idx, :, 0])

    def update_mask(self, mask, dynamic, chosen_idx):

        assert mask.shape[0] == dynamic.shape[0] == chosen_idx.shape[0] == 1

        # Chosen to dropoff the driver / terminate the route
        if chosen_idx[0] == 0:
            return mask * 0

        # If carpool is maxed, we must drop off all current riders
        if torch.sum(dynamic.data != 0) >= self.max_num_seats:
            mask = (dynamic[:, 0] == 1).type(torch.FloatTensor)

        # Dropoff all current riders before we dropoff the driver
        if torch.sum(dynamic.data == 1) > 1:
            mask[:, 0] = 0
        else:
            mask[:, 0] = 1

        # If we've picked a node twice, we can't pick it again
        if dynamic[0, 0, chosen_idx[0]].data.numpy() == -1:
            mask[0, chosen_idx[0]] = 0
        return mask

    @staticmethod
    def update_dynamic(dynamic, chosen_idx):

        # If the rider is inactive (0), make it active (1)
        # If the rider is active (1), make it unavailable (-1)
        if dynamic[0, 0, chosen_idx[0]].data.numpy() == 0:
            dynamic[0, 0, chosen_idx[0]] = 1
        else:
            dynamic[0, 0, chosen_idx[0]] = -1
        return dynamic

    def reward(self, static, tour_indices, use_cuda=False):
        """
        Function of: tour_length + number_passengers
        """

        num_unique = len(np.unique(tour_indices))
        assert num_unique <= self.max_num_seats + 1, 'Invalid tour: %s' % np.unique(tour_indices)
        assert tour_indices[0, -1] == 0, 'Must end with the drivers node'

        idx = tour_indices[0].numpy().tolist()
        counts = {a: idx.count(a) for a in np.unique(tour_indices)}
        assert all(v <= 2 for _, v in counts.items()), 'Can only select a max of 2 times'

        tour_len = Variable(torch.FloatTensor([0]))

        seen = set([0])
        prev_val = static[0, :2, 0]
        for idx in tour_indices[0]:

            if idx not in seen:
                cur_val = static[0, :2, idx]
                seen.add(idx)
            else:
                cur_val = static[0, 2:, idx]

            tour_len = tour_len + torch.sqrt(torch.sum(torch.pow(prev_val - cur_val, 2)))
            prev_val = cur_val

        orig_len = torch.sum(torch.pow(static[0, :2, 0] - static[0, 2:, 0], 2))
        orig_len = torch.sqrt(orig_len)

        num_passengers = (num_unique - 1) / float(self.max_num_seats)

        diff = tour_len - 2. * orig_len
        if diff.data.numpy() > 0:
            penalty = 10.
        else:
            penalty = 0

        tour = tour_len - orig_len - num_passengers + penalty

        return tour.unsqueeze(0)

    @staticmethod
    def render(static, tour_indices):
        import matplotlib.pyplot as plt

        plt.close('all')

        data = static.data.numpy().copy()[0]
        indices = tour_indices.squeeze().numpy()
        num_plots = min(int(np.sqrt(len(tour_indices))), 3)

        print('indices: ', indices)

        for i in range(num_plots):
            for j in range(num_plots):

                plt.subplot(num_plots, num_plots, i * num_plots + j + 1)

                seen = set([0])
                x = [data[0, 0]]
                y = [data[1, 0]]

                for idx in indices:

                    if idx not in seen:
                        x.append(data[0, idx])
                        y.append(data[1, idx])
                        seen.add(idx)
                    else:
                        x.append(data[2, idx])
                        y.append(data[3, idx])

                plt.plot(x, y)

                all_x = np.hstack((data[0], data[2]))
                all_y = np.hstack((data[1], data[3]))
                plt.scatter(all_x, all_y, s=4, c='r')
        plt.tight_layout()


if __name__ == '__main__':

    import sys
    sys.path.append('..')
    from model import DRL4VRP

    task = 'carpool'
    train_size = 100000
    val_size = 1000
    batch_size = 1
    static_size = 4
    dynamic_size = 1
    hidden_size = 128
    dropout = 0.3
    use_cuda = False
    num_layers = 1
    critic_beta = 0.9
    max_grad_norm = 2.
    actor_lr = 1e-4
    actor_decay_step = 5000
    actor_decay_rate = 0.96
    plot_every = 500

    model = DRL4VRP(static_size, dynamic_size, hidden_size, dropout,
                    num_layers, critic_beta, max_grad_norm,
                    actor_lr, actor_decay_step, actor_decay_rate, plot_every, use_cuda)

    for epoch in range(100):

        num_nodes = 50
        train = ConstDemandCarpoolDataset(100000, num_nodes, max_num_seats=5)
        valid = ConstDemandCarpoolDataset(1000, num_nodes, max_num_seats=5)

        reward_fn = train.reward
        model.train(train, valid, reward_fn, batch_size)
