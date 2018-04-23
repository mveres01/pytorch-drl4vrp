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
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import DRL4VRP, Encoder, Attention


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the input space, and tries to
    estimate how long it thinks a tour will be.
    """

    def __init__(self, static_size, dynamic_size, hidden_size,
                 num_process_iter, use_cuda):
        super(Critic, self).__init__()

        # How many times we want to look at the input & update the context vec
        self.num_process_iter = num_process_iter
        self.use_cuda = use_cuda

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.attn = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 20)
        self.fc2 = nn.Linear(20, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform(p)

    def forward(self, static, dynamic, initial_state):

        static_enc = self.static_encoder(static)
        dynamic_enc = self.dynamic_encoder(dynamic)

        batch_size, num_feats, _ = static_enc.size()

        if initial_state is None:
            # Use an empty context vector
            if self.use_cuda:
                x0 = torch.cuda.FloatTensor(1, batch_size, num_feats).fill_(0)
            else:
                x0 = torch.FloatTensor(1, batch_size, num_feats).fill_(0)
            context = Variable(x0)
        else:
            # Pass initial state through an encoder
            context = self.static_encoder(initial_state.unsqueeze(2))
            context = context.permute(2, 0, 1)

        static_p = static_enc.permute(0, 2, 1)
        for _ in range(self.num_process_iter):

            # Attention is applied across the static and dynamic states
            attn = self.attn(static_enc, dynamic_enc, context)  # (B, 1, Seq)
            context[0] = attn.bmm(static_p).squeeze(1)

        output = F.relu(self.fc1(context.squeeze(0)))
        output = self.fc2(output)

        return F.elu(output)


def validate(data_loader, actor, reward_fn, render_fn, save_dir, use_cuda):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    rewards, indices, inputs = [], [], []

    for batch_idx, batch in enumerate(data_loader):

        static = Variable(batch[0].cuda() if use_cuda else batch[0])
        dynamic = Variable(batch[1].cuda() if use_cuda else batch[1])

        if len(batch[2]) > 0:
            initial_state = Variable(batch[2].cuda() if use_cuda else batch[2])
        else:
            initial_state = None

        # Full forward pass through the dataset
        tour_indices, _ = actor.forward(static, dynamic, initial_state)

        reward = reward_fn(static, tour_indices, use_cuda)

        # GOALS: TSP_20=3.97, TSP_50=6.08, TSP_100=8.44
        rewards.append(torch.mean(reward.data))

        if batch_idx < 50:
            indices.append(tour_indices)
            inputs.append(static)

    inputs = torch.cat(inputs, dim=0)

    mean_reward = np.mean(rewards)

    if render_fn is not None:
        save_path = os.path.join(save_dir, '%2.4f_valid.png' % mean_reward)
        render_fn(inputs, indices, save_path)

    actor.train()

    return mean_reward


def train(actor, critic, problem, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr,
          max_grad_norm, plot_every, checkpoint_every, use_cuda):
    """Constructs the main actor & critic networks, and performs all training."""

    save_dir = os.path.join(problem, '%d' % num_nodes)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    for epoch in range(100):

        train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
        valid_loader = DataLoader(valid_data, 1, True, num_workers=0)

        actor.train()
        critic.train()

        losses, rewards = [], []
        for batch_idx, batch in enumerate(train_loader):

            start = time.time()

            static = Variable(batch[0].cuda() if use_cuda else batch[0])
            dynamic = Variable(batch[1].cuda() if use_cuda else batch[1])

            if len(batch[2]) > 0:
                initial_state = batch[2]
                if use_cuda:
                    initial_state = initial_state.cuda()
                initial_state = Variable(initial_state)
            else:
                initial_state = None

            # Full forward pass through the dataset
            tour_indices, logp_tour = actor.forward(static, dynamic,
                                                    initial_state)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices, use_cuda)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic, initial_state).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage * logp_tour)
            critic_loss = torch.mean(torch.pow(advantage, 2))

            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(actor.parameters(),
                                          max_grad_norm, norm_type=2)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm(critic.parameters(),
                                          max_grad_norm, norm_type=2)
            critic_optim.step()

            # GOALS: TSP_20=3.97, TSP_50=6.08, TSP_100=8.44
            rewards.append(torch.mean(reward.data))
            losses.append(torch.mean(actor_loss.data))
            if (render_fn is not None) and (batch_idx + 1) % plot_every == 0:

                save_path = os.path.join(save_dir, '%d.png' % batch_idx)
                render_fn(static, tour_indices, save_path)

            if (batch_idx + 1) % checkpoint_every == 0:

                mean_loss = np.mean(losses[-checkpoint_every:])
                mean_reward = np.mean(rewards[-checkpoint_every:])

                print('%d/%d, avg. reward: %2.4f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader),
                       mean_reward, mean_loss, time.time() - start))

                fname = 'batch%d_%2.4f_actor.pt' % (batch_idx, mean_reward)
                save_path = os.path.join(checkpoint_dir, fname)
                torch.save(actor.state_dict(), save_path)

                fname = 'batch%d_%2.4f_critic.pt' % (batch_idx, mean_reward)
                save_path = os.path.join(checkpoint_dir, fname)
                torch.save(critic.state_dict(), save_path)

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        mean_valid = validate(valid_loader, actor, reward_fn, render_fn, save_dir, use_cuda)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f' % (mean_loss, mean_reward, mean_valid))


def train_tsp():

    # TSP20, 3.82  (Optimal) - 3.97  (DRL4VRP)
    # TSP50, 5.70  (Optimal) - 6.08  (DRL4VRP)
    # TSP100, 7.77 (OptimalBS) - 8.44 (DRL4VRP)

    from tasks import tsp
    from tasks.tsp import TSPDataset

    kwargs = {}

    train_size = 1000000
    valid_size = 1000
    num_nodes = 10

    train_data = TSPDataset(size=num_nodes, num_samples=train_size)
    valid_data = TSPDataset(size=num_nodes, num_samples=valid_size)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tsp.reward
    kwargs['render_fn'] = tsp.render
    kwargs['problem'] = 'tsp'
    mask_fn = tsp.update_mask
    update_fn = None

    static_size = 2
    dynamic_size = 1
    hidden_size = 128
    dropout = 0.1
    num_layers = 1
    use_cuda = torch.cuda.is_available()
    num_process_iter = 3

    actor = DRL4VRP(static_size, dynamic_size, hidden_size, update_fn,
                    mask_fn, dropout, num_layers, use_cuda)
    critic = Critic(static_size, dynamic_size, hidden_size, num_process_iter,
                    use_cuda)

    if use_cuda:
        actor.cuda()
        critic.cuda()

    kwargs['num_nodes'] = num_nodes
    kwargs['batch_size'] = 64
    kwargs['actor_lr'] = 5e-4
    kwargs['critic_lr'] = kwargs['actor_lr']
    kwargs['max_grad_norm'] = 2.
    kwargs['plot_every'] = 500
    kwargs['checkpoint_every'] = 100
    kwargs['use_cuda'] = use_cuda

    train(actor, critic, **kwargs)


def train_vrp():

    # VRP10, Capacity 20:  4.65  (BS) - 4.80  (Greedy)
    # VRP20, Capacity 30:  6.34  (BS) - 6.51  (Greedy)
    # VRP50, Capacity 40:  11.08 (BS) - 11.32 (Greedy)
    # VRP100, Capacity 50: 16.86 (BS) - 17.12 (Greedy)

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset

    kwargs = {}

    train_size = 100000
    valid_size = 1000
    num_nodes = 10
    max_load = 20
    max_demand = 9

    train_data = VehicleRoutingDataset(train_size, num_nodes, max_load, max_demand)
    valid_data = VehicleRoutingDataset(valid_size, num_nodes, max_load, max_demand)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render
    kwargs['problem'] = 'vrp'
    mask_fn = train_data.update_mask
    update_fn = train_data.update_dynamic

    static_size = 2
    dynamic_size = 2
    hidden_size = 128
    dropout = 0.1
    num_layers = 1
    use_cuda = torch.cuda.is_available()
    num_process_iter = 3

    actor = DRL4VRP(static_size, dynamic_size, hidden_size, update_fn,
                    mask_fn, dropout, num_layers, use_cuda)
    critic = Critic(static_size, dynamic_size, hidden_size, num_process_iter,
                    use_cuda)

    if use_cuda:
        actor.cuda()
        critic.cuda()

    kwargs['num_nodes'] = num_nodes
    kwargs['batch_size'] = 64
    kwargs['actor_lr'] = 5e-4
    kwargs['critic_lr'] = kwargs['actor_lr']
    kwargs['max_grad_norm'] = 2.
    kwargs['plot_every'] = 500
    kwargs['checkpoint_every'] = 100
    kwargs['use_cuda'] = use_cuda

    #kwargs['actor'] = actor
    #kwargs['critic'] = critic

    train(actor, critic, **kwargs)


if __name__ == '__main__':
    # train_vrp()
    train_tsp()
