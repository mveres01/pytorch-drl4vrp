# code based in part on
# http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
# and from
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py

#  NOTE: File from: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/blob/master/tsp_task.py

import time
import requests
from tqdm import tqdm
import os
import numpy as np
import re
import zipfile
import itertools
from collections import namedtuple
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from model import DRL4VRP, Critic
from utils import gen_dataset

#######################################
# Functions for downloading dataset
#######################################
TSP = namedtuple('TSP', ['x', 'y', 'name'])

GOOGLE_DRIVE_IDS = {
    'tsp5_train.zip': '0B2fg8yPGn2TCSW1pNTJMXzFPYTg',
    'tsp10_train.zip': '0B2fg8yPGn2TCbHowM0hfOTJCNkU',
    'tsp5-20_train.zip': '0B2fg8yPGn2TCTWNxX21jTDBGeXc',
    'tsp50_train.zip': '0B2fg8yPGn2TCaVQxSl9ab29QajA',
    'tsp20_test.txt': '0B2fg8yPGn2TCdF9TUU5DZVNCNjQ',
    'tsp40_test.txt': '0B2fg8yPGn2TCcjFrYk85SGFVNlU',
    'tsp50_test.txt.zip': '0B2fg8yPGn2TCUVlCQmQtelpZTTQ',
}


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    return True


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_google_drive_file(data_dir, task, min_length, max_length):
    paths = {}
    for mode in ['train', 'test']:
        candidates = []
        candidates.append(
            '{}{}_{}'.format(task, max_length, mode))
        candidates.append(
            '{}{}-{}_{}'.format(task, min_length, max_length, mode))

        for key in candidates:
            print(key)
            for search_key in GOOGLE_DRIVE_IDS.keys():
                if search_key.startswith(key):
                    path = os.path.join(data_dir, search_key)
                    print("Download dataset of the paper to {}".format(path))

                    if not os.path.exists(path):
                        download_file_from_google_drive(GOOGLE_DRIVE_IDS[search_key], path)
                    if path.endswith('zip'):
                        with zipfile.ZipFile(path, 'r') as z:
                            z.extractall(data_dir)
                    paths[mode] = path

    return paths


def read_paper_dataset(paths, max_length):
    x, y = [], []
    for path in paths:
        print("Read dataset {} which is used in the paper..".format(path))
        length = max(re.findall('\d+', path))
        with open(path) as f:
            for l in tqdm(f):
                inputs, outputs = l.split(' output ')
                x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]))
                y.append(np.array(outputs.split(), dtype=np.int32)[:-1])  # skip the last one

    return x, y


def maybe_generate_and_save(self, except_list=[]):
    data = {}

    for name, num in self.data_num.items():
        if name in except_list:
            print("Skip creating {} because of given except_list {}".format(name, except_list))
            continue
        path = self.get_path(name)

        print("Skip creating {} for [{}]".format(path, self.task))
        tmp = np.load(path)
        self.data[name] = TSP(x=tmp['x'], y=tmp['y'], name=name)


def get_path(self, name):
    return os.path.join(
        self.data_dir, "{}_{}={}.npz".format(
            self.task_name, name, self.data_num[name]))


def read_zip_and_update_data(self, path, name):
    if path.endswith('zip'):
        filenames = zipfile.ZipFile(path).namelist()
        paths = [os.path.join(self.data_dir, filename) for filename in filenames]
    else:
        paths = [path]

    x_list, y_list = read_paper_dataset(paths, self.max_length)

    x = np.zeros([len(x_list), self.max_length, 2], dtype=np.float32)
    y = np.zeros([len(y_list), self.max_length], dtype=np.int32)

    for idx, (nodes, res) in enumerate(tqdm(zip(x_list, y_list))):
        x[idx, :len(nodes)] = nodes
        y[idx, :len(res)] = res

    if self.data is None:
        self.data = {}

    print("Update [{}] data with {} used in the paper".format(name, path))
    self.data[name] = TSP(x=x, y=y, name=name)


def create_dataset(
        problem_size,
        data_dir):

    def find_or_return_empty(data_dir, problem_size):

        val_fname1 = os.path.join(data_dir, 'tsp{}_test.txt'.format(problem_size))
        val_fname2 = os.path.join(data_dir, 'tsp-{}_test.txt'.format(problem_size))

        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        else:
            if os.path.exists(val_fname1):
                return val_fname1
            elif os.path.exists(val_fname2):
                return val_fname2
        return None

    val = find_or_return_empty(data_dir, problem_size)
    if val is None:
        download_google_drive_file(data_dir, 'tsp', '', problem_size)
        val = find_or_return_empty(data_dir, problem_size)

    return val


class TSPDataset(Dataset):

    def __init__(self, fname=None, train=False, size=50, num_samples=1e6, seed=1234):
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

    @staticmethod
    def update_mask(mask, dynamic, chosen_idx):

        assert mask.shape[0] == dynamic.shape[0] == len(chosen_idx)

        mask[np.arange(dynamic.size(0)), chosen_idx] = 0
        return mask

    @staticmethod
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
        y = torch.cat((tour, tour[:, 0:1]), dim=1)

        # Euclidean distance between each consecutive point
        tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

        return Variable(tour_len).sum(1)

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
            data = torch.gather(static[i].data, 1, idx).cpu().numpy()

            plt.subplot(num_plots, num_plots, i + 1)
            plt.plot(data[0], data[1], zorder=1)
            plt.scatter(data[0], data[1], s=4, c='r', zorder=2)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=400)
