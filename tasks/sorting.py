# Generate sorting data and store in .txt
# Define the reward function
# Taken from: https://raw.githubusercontent.com/pemami4911/neural-combinatorial-rl-pytorch/master/sorting_task.py

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import trange, tqdm
import os
import sys


def reward(sample_solution, USE_CUDA=False):
    """
    The reward for the sorting task is defined as the
    length of the longest sorted consecutive subsequence.

    Input sequences must all be the same length.

    Example:

    input       | output
    ====================
    [1 4 3 5 2] | [5 1 2 3 4]

    The output gets a reward of 4/5, or 0.8

    The range is [1/sourceL, 1]

    Args:
        sample_solution: list of len sourceL of [batch_size]
        Tensors
    Returns:
        [batch_size] containing trajectory rewards
    """
    batch_size = sample_solution[0].size(0)
    sourceL = len(sample_solution)

    longest = Variable(torch.ones(batch_size), requires_grad=False)
    current = Variable(torch.ones(batch_size), requires_grad=False)

    if USE_CUDA:
        longest = longest.cuda()
        current = current.cuda()

    for i in range(1, sourceL):
        # compare solution[i-1] < solution[i]
        res = torch.lt(sample_solution[i - 1], sample_solution[i])
        # if res[i,j] == 1, increment length of current sorted subsequence
        current += res.float()
        # else, reset current to 1
        current[torch.eq(res, 0)] = 1
        # current[torch.eq(res, 0)] -= 1
        # if, for any, current > longest, update longest
        mask = torch.gt(current, longest)
        longest[mask] = current[mask]
    return -torch.div(longest, sourceL)


def create_dataset(num_samples, data_dir, data_len, prefix='train', seed=None):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if seed is not None:
        np.random.seed(seed)

    task = 'sorting-size-{}-len-{}-{}.txt'.format(num_samples, data_len, prefix)

    task_path = os.path.join(data_dir, task)
    if os.path.exists(task_path):
        return task_path

    print('Creating training data set for {}...'.format(task))

    with open(task_path, 'w') as f:
        for _ in range(num_samples):
            x = np.random.permutation(data_len)
            f.write(''.join(['%d ' % d for d in x])[:-1] + '\n')
    return task_path


class SortingDataset(Dataset):

    def __init__(self, dataset_fname):
        super(SortingDataset, self).__init__()

        print('Loading training data into memory')
        self.dataset = []
        with open(dataset_fname, 'r') as dset:
            lines = dset.readlines()
            for next_line in tqdm(lines):
                toks = next_line.split()
                sample = torch.zeros(1, len(toks)).long()
                for idx, tok in enumerate(toks):
                    sample[0, idx] = int(tok)
                self.dataset.append(sample)

        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx].transpose(1, 0)


if __name__ == '__main__':
    if int(sys.argv[1]) == 0:
        # sample = Variable(torch.Tensor([[3, 2, 1, 4, 5], [2, 3, 5, 1, 4]]))
        sample = [Variable(torch.Tensor([3, 2])), Variable(torch.Tensor([2, 3])), Variable(torch.Tensor([1, 5])),
                  Variable(torch.Tensor([4, 1])), Variable(torch.Tensor([5, 4]))]
        answer = torch.Tensor([3 / 5., 3 / 5])

        res = reward(sample)

        print('Expected answer: {}, Actual answer: {}'.format(answer, res.data))
        """
        sample = Variable(torch.Tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]))
        answer = torch.Tensor([1., 1/5])

        res = reward(sample)

        print('Expected answer: {}, Actual answer: {}'.format(answer, res.data))

        sample = Variable(torch.Tensor([[1, 2, 5, 4, 3], [4, 1, 2, 3, 5]]))
        answer = torch.Tensor([3/5., 4/5])

        res = reward(sample)

        print('Expected answer: {}, Actual answer: {}'.format(answer, res.data))
        """
    elif int(sys.argv[1]) == 1:
        create_sorting_dataset(1000, 100, 'data', 10, 123)
    elif int(sys.argv[1]) == 2:

        sorting_data = SortingDataset('data', 'sorting-size-1000-len-10-train.txt',
                                      'sorting-size-100-len-10-val.txt')

        for i in range(len(sorting_data)):
            print(sorting_data[i])
