import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: Have the "demand" of the depot dynamically change when an index is selected,
# so the demand is not always constant***


class VehicleRoutingDataset(Dataset):

    def __init__(self, num_samples, input_size, max_load=20, max_demand=9, seed=1234):
        super(VehicleRoutingDataset, self).__init__()

        assert max_load > max_demand
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.input_size = input_size
        self.max_load = max_load
        self.max_demand = max_demand

        # Driver location will be the first node in each
        locations = np.random.uniform(0, 1, (num_samples, 2, input_size + 1))
        self.static = torch.FloatTensor(locations)

        # Vehicle needs a load > 0 which gets broadcasted to all states
        loads = np.full((num_samples, 1, input_size + 1), max_load) / float(max_load)

        # All nodes are assigned a random demand in [1, max_demand]
        demands = np.random.randint(1, max_demand + 1, (num_samples, 1, input_size + 1))
        demands = demands / float(max_load)

        # The depot will be used to refill the vehicle with the initial load
        demands[:, 0, 0] = -loads[:, 0, 0]
        self.dynamic = torch.FloatTensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Note that all math is done using integers to avoid float errors

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """

        assert mask.shape[0] == dynamic.shape[0] == 1

        dynamic_int = (dynamic.data * self.max_load).int()

        # Nodes with 0-demand cannot be chosen
        mask = dynamic_int[:, 1] > 0

        if not mask[0:, 1:].byte().any():  # no demand, terminate search
            mask = torch.zeros_like(mask)
        elif dynamic_int[0, 0, 0] <= 0:  # vehicle has no load -> refill
            mask = torch.zeros_like(mask)
            mask[:, 0] = 1
        return Variable(mask.float())

    def update_dynamic(self, dynamic_var, chosen_idx):

        to_update = chosen_idx[0]

        # Clone the dynamic variable so we don't mess up graph
        dynamic = dynamic_var.clone()

        # Agent chooses to return to the depot, so we reset the 'vehicle load'
        # for each dynamic element to be the original capacity
        if to_update == 0:
            dynamic.data[0, 0, :] = -dynamic.data[0, 1, 0]
        else:
            dint = (dynamic.data * self.max_load).int()

            load = dint[0, 0, to_update]
            demand = dint[0, 1, to_update]

            load_t = np.maximum(0, load - demand) / self.max_load
            demand_t = np.maximum(0, demand - load) / self.max_load

            dynamic.data[0, 0, :] = load_t
            dynamic.data[0, 1, to_update] = demand_t

        return dynamic

    @staticmethod
    def reward(static, tour_indices, use_cuda=False):
        """
        Function of: tour_length + number_passengers
        """

        dummy = torch.FloatTensor([0])
        tour_len = Variable(dummy.cuda() if use_cuda else dummy)

        # Start at the depot
        prev_loc = static[0, : 2, 0]
        for idx in tour_indices[0]:

            cur_loc = static[0, : 2, idx]
            tour_len = tour_len + torch.sqrt(torch.sum(torch.pow(prev_loc - cur_loc, 2)))
            prev_loc = cur_loc

        # End at the depot
        tour_len = tour_len + torch.sqrt(torch.sum(torch.pow(prev_loc - static[0, : 2, 0], 2)))

        return tour_len.unsqueeze(0)

    @staticmethod
    def render(static, tour_indices, save_path):

        plt.close('all')

        num_plots = min(int(np.sqrt(len(tour_indices))), 3)

        for i in range(num_plots ** 2):

            # Convert the indices back into a tour
            idx = tour_indices[i]
            if len(idx.size()) == 1:
                idx = idx.unsqueeze(0)

            idx = idx.expand(static.size(1), -1)
            data = torch.gather(static[i].data, 1, idx).numpy()

            start = static[i, :, 0].data.numpy()
            x = np.hstack((start[0], data[0], start[0]))
            y = np.hstack((start[1], data[1], start[1]))

            plt.subplot(num_plots, num_plots, i + 1)

            # Assign each subtour a different colour & label in order traveled
            idx = np.hstack((0, idx.numpy().flatten(), 0))
            where = np.where(idx == 0)[0]
            count = 0
            for j in range(len(where) - 1):

                count += 1
                low = where[j]
                high = where[j + 1]

                plt.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=count)

            plt.legend(loc="upper right", fontsize=3, framealpha=0.5)
            plt.scatter(x, y, s=4, c='r', zorder=2)
            plt.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=400)
