import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=1234):
        super(TSPDataset, self).__init__()

        torch.manual_seed(seed)

        self.dataset = torch.FloatTensor(num_samples, 2, size).uniform_(0, 1)
        self.dynamic = torch.zeros(num_samples, 1, size)
        self.num_nodes = size
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.dataset[idx], self.dynamic[idx], [])


def update_mask(mask, dynamic, chosen_idx):
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, tour_indices, use_cuda=False):
    """
    Parameters
    ----------
    tour: torch.FloatTensor of size (batch_size, num_features, seq_len)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route of size (batch_size, seq_len)
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return Variable(tour_len).sum(1)


def render(static, tour_indices, save_path):

    plt.close('all')

    num_plots = min(int(np.sqrt(len(tour_indices))), 3)

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
