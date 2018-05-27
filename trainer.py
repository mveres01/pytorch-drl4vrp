"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DRL4TSP, Encoder, Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by 
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, reward_fn, render_fn, save_dir):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    rewards = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, _ = actor.forward(static, dynamic, x0)

            reward = reward_fn(static, tour_indices)
            rewards.append(torch.mean(reward.detach()))

            if render_fn is not None and batch_idx < 50:
                mean_reward = np.mean(rewards)
                save_path = os.path.join(save_dir, '%2.4f_valid.png' % mean_reward)
                render_fn(static, tour_indices, save_path)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, problem, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, checkpoint_every):
    """Constructs the main actor & critic networks, and performs all training."""

    save_dir = os.path.join(problem, '%d' % num_nodes)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    for epoch in range(100):

        actor.train()
        critic.train()

        losses, rewards, critic_rewards = [], [], []
        for batch_idx, batch in enumerate(train_loader):

            start = time.time()

            static, dynamic, x0 = batch

            static = static.to(device).requires_grad_()
            dynamic = dynamic.to(device).requires_grad_()
            x0 = x0.to(device).requires_grad_() if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_in = torch.tensor(tour_logp.data, device=device, requires_grad=True)
            critic_est = critic(critic_in).squeeze(1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(torch.pow(advantage, 2))

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            # GOALS: TSP_20=3.97, TSP_50=6.08, TSP_100=8.44
            critic_rewards.append(torch.mean(critic_est.detach().data))
            rewards.append(torch.mean(reward.detach().data))
            losses.append(torch.mean(actor_loss.detach().data))

            if (batch_idx + 1) % checkpoint_every == 0:

                mean_loss = np.mean(losses[-checkpoint_every:])
                mean_reward = np.mean(rewards[-checkpoint_every:])
                mean_critic_reward = np.mean(critic_rewards[-checkpoint_every:])

                prefix = 'epoch%dbatch%d_%2.4f' % (epoch, batch_idx, mean_reward)
                save_path = os.path.join(checkpoint_dir, prefix + '_actor.pt')
                torch.save(actor.state_dict(), save_path)

                save_path = os.path.join(checkpoint_dir, prefix + '_critic.pt')
                torch.save(critic.state_dict(), save_path)

                if render_fn is not None:
                    save_path = os.path.join(save_dir, 'epoch%d_%d.png' %
                                             (epoch, batch_idx))
                    render_fn(static, tour_indices, save_path)

                print('%d/%d, reward: %2.3f, pred: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader),
                       mean_reward, mean_critic_reward,  mean_loss, time.time() - start))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        mean_valid = validate(valid_loader, actor, reward_fn, render_fn, save_dir)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f' % (mean_loss, mean_reward, mean_valid))


def train_tsp():

    # TSP20, 3.82  (Optimal) - 3.97  (DRL4VRP)
    # TSP50, 5.70  (Optimal) - 6.08  (DRL4VRP)
    # TSP100, 7.77 (OptimalBS) - 8.44 (DRL4VRP)

    from tasks import tsp
    from tasks.tsp import TSPDataset

    train_size = 1000000
    valid_size = 1000
    num_nodes = 50
    train_data = TSPDataset(size=num_nodes, num_samples=train_size)
    valid_data = TSPDataset(size=num_nodes, num_samples=valid_size)

    # Model - specific parameters
    static_size = 2
    dynamic_size = 1
    hidden_size = 128
    dropout = 0.1
    num_layers = 1

    kwargs = {}
    kwargs['problem'] = 'tsp'
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['num_nodes'] = num_nodes
    kwargs['batch_size'] = 2
    kwargs['actor_lr'] = 1e-3
    kwargs['critic_lr'] = kwargs['actor_lr']
    kwargs['max_grad_norm'] = 2.
    kwargs['checkpoint_every'] = 1000
    kwargs['reward_fn'] = tsp.reward
    kwargs['render_fn'] = tsp.render
    mask_fn = tsp.update_mask
    update_fn = None

    actor = DRL4TSP(static_size, dynamic_size, hidden_size, update_fn,
                    mask_fn, num_layers, dropout).to(device)
    critic = Critic(hidden_size).to(device)

    train(actor, critic, **kwargs)


def train_vrp():

    # VRP10, Capacity 20:  4.65  (BS) - 4.80  (Greedy)
    # VRP20, Capacity 30:  6.34  (BS) - 6.51  (Greedy)
    # VRP50, Capacity 40:  11.08 (BS) - 11.32 (Greedy)
    # VRP100, Capacity 50: 16.86 (BS) - 17.12 (Greedy)

    CAPACITY_DICT = {10: 20, 20: 30, 50: 40, 100: 50}

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset

    # Problem - specific parameters
    train_size = 1000000
    valid_size = 1000
    max_demand = 9
    num_nodes = 50
    max_load = CAPACITY_DICT[num_nodes]
    train_data = VehicleRoutingDataset(train_size, num_nodes, max_load, max_demand)
    valid_data = VehicleRoutingDataset(valid_size, num_nodes, max_load, max_demand)

    # Model - specific parameters
    static_size = 2
    dynamic_size = 2 # (load, demand)
    hidden_size = 128
    dropout = 0.1
    num_layers = 1

    kwargs = {}
    kwargs['problem'] = 'vrp'
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render
    kwargs['num_nodes'] = num_nodes
    kwargs['batch_size'] = 200
    kwargs['actor_lr'] = 1e-3
    kwargs['critic_lr'] = kwargs['actor_lr']
    kwargs['max_grad_norm'] = 2.
    kwargs['checkpoint_every'] = 1000

    mask_fn = train_data.update_mask
    update_fn = train_data.update_dynamic

    actor = DRL4TSP(static_size, dynamic_size, hidden_size, update_fn,
                    mask_fn, num_layers, dropout).to(device)
    critic = Critic(hidden_size).to(device)

    train(actor, critic, **kwargs)

    '''
    # path = 'vrp/50/checkpoints/batch13499_11.5925_'
    path = 'vrp/50/checkpoints/batch499_11.6461_'
    params = torch.load(path + 'actor.pt', map_location=lambda storage, loc: storage)
    actor.load_state_dict(params)

    params = torch.load(path + 'critic.pt', map_location=lambda storage, loc: storage)
    critic.load_state_dict(params)
    '''


if __name__ == '__main__':
    train_vrp()
    # train_tsp()
