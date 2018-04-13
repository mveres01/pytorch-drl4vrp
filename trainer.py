import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import DRL4TSP, DRL4VRP, Encoder, Attention


class Critic(nn.Module):
    """Estimates the problem complexity."""

    def __init__(self, static_size, dynamic_size, hidden_size, num_process_iter,
                 use_cuda):
        super(Critic, self).__init__()

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


class NeuralCombinatorialSolver(object):

    def __init__(self, actor, critic, reward_fn, render_fn, 
                 plot_every=10, checkpoint_every=500, save_dir='outputs',
                 use_cuda=False):

        self.actor = actor
        self.critic = critic
        self.reward_fn = reward_fn
        self.render_fn = render_fn
        self.plot_every = plot_every
        self.checkpoint_every = checkpoint_every
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.use_cuda = use_cuda

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self, data_loader, actor_lr, critic_lr, max_grad_norm):

        self.actor.train()
        self.critic.train()

        actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        losses, rewards = [], []
        for batch_idx, batch in enumerate(data_loader):

            gc.collect()
            start = time.time()

            static = Variable(batch[0].cuda() if self.use_cuda else batch[0])
            dynamic = Variable(batch[1].cuda() if self.use_cuda else batch[1])

            if len(batch[2]) > 0:
                initial_state = batch[2]
                if self.use_cuda:
                    initial_state = initial_state.cuda()
                initial_state = Variable(initial_state)
            else:
                initial_state = None

            # Full forward pass through the dataset
            tour_indices, logp_tour = self.actor.forward(static, dynamic,
                                                         initial_state)

            # Sum the log probabilities for each city in the tour
            reward = self.reward_fn(static, tour_indices, self.use_cuda)

            # Query the critic for an estimate of the reward
            critic_est = self.critic(static, dynamic, initial_state).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage * logp_tour)
            critic_loss = torch.mean(torch.pow(advantage, 2))

            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(self.actor.parameters(),
                                          max_grad_norm, norm_type=2)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic.parameters(),
                                          max_grad_norm, norm_type=2)
            critic_optimizer.step()

            # GOALS: TSP_20=3.97, TSP_50=6.08, TSP_100=8.44
            rewards.append(torch.mean(reward.data))
            losses.append(torch.mean(actor_loss.data))
            if (batch_idx + 1) % self.plot_every == 0:

                save_path = os.path.join(self.save_dir, '%d.png' % batch_idx)
                self.render_fn(static, tour_indices, save_path)

            if (batch_idx + 1) % self.checkpoint_every == 0:

                mean_loss = np.mean(losses[-self.checkpoint_every:])
                mean_reward = np.mean(rewards[-self.checkpoint_every:])

                print('%d/%d, avg. reward: %2.4f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(data_loader),
                       mean_reward, mean_loss, time.time() - start))

                fname = 'batch%d_%2.4f_actor.pt' % (batch_idx, mean_reward)
                save_path = os.path.join(self.checkpoint_dir, fname)
                torch.save(self.actor.state_dict(), save_path)

                fname = 'batch%d_%2.4f_critic.pt' % (batch_idx, mean_reward)
                save_path = os.path.join(self.checkpoint_dir, fname)
                torch.save(self.critic.state_dict(), save_path)

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        return mean_reward, mean_loss

    def eval(self, data_loader):

        self.actor.eval()

        rewards = []
        indices = []
        inputs = []

        for batch_idx, batch in enumerate(data_loader):

            gc.collect()

            static = Variable(batch[0].cuda() if self.use_cuda else batch[0])
            dynamic = Variable(batch[1].cuda() if self.use_cuda else batch[1])

            if len(batch[2]) > 0:
                initial_state = batch[2]
                if self.use_cuda:
                    initial_state = initial_state.cuda()
                initial_state = Variable(initial_state)
            else:
                initial_state = None

            # Full forward pass through the dataset
            tour_indices, _ = self.actor.forward(static, dynamic,
                                                 initial_state)

            reward = self.reward_fn(static, tour_indices, self.use_cuda)

            # GOALS: TSP_20=3.97, TSP_50=6.08, TSP_100=8.44
            rewards.append(torch.mean(reward.data))

            if batch_idx < 50:
                indices.append(tour_indices)
                inputs.append(static)

        mean_reward = np.mean(rewards)

        inputs = torch.cat(inputs, dim=0)

        save_path = os.path.join(self.save_dir, '%2.4f_valid.png' % mean_reward)
        self.render_fn(inputs, indices, save_path)

        return mean_reward


def train_tsp():

    # TSP20, 3.82  (Optimal) - 3.97  (DRL4VRP)
    # TSP50, 5.70  (Optimal) - 6.08  (DRL4VRP)
    # TSP100, 7.77 (OptimalBS) - 8.44 (DRL4VRP)

    from tasks import tsp
    from tasks.tsp import TSPDataset

    num_nodes = 10
    save_dir = 'tsp_outputs/tsp_%s' % num_nodes

    train_size = 1000000
    val_size = 1000
    batch_size = 64
    num_process_iter = 3
    static_size = 2
    dynamic_size = 1
    hidden_size = 128
    dropout = 0.2
    use_cuda = torch.cuda.is_available()
    num_layers = 1
    max_grad_norm = 2.
    actor_lr = 1e-3
    critic_lr = actor_lr
    plot_every = 250
    checkpoint_every = 500

    train_data = TSPDataset(size=num_nodes, num_samples=train_size)
    valid_data = TSPDataset(size=num_nodes, num_samples=val_size)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, 1, False, num_workers=0)

    mask_fn = tsp.update_mask
    update_fn = None
    reward_fn = tsp.reward
    render_fn = tsp.render

    actor = DRL4TSP(static_size, dynamic_size, hidden_size, update_fn, mask_fn,
                    dropout, num_layers, use_cuda)

    critic = Critic(static_size, dynamic_size, hidden_size, num_process_iter, use_cuda)

    if use_cuda:
        actor.cuda()
        critic.cuda()

    solver = NeuralCombinatorialSolver(actor, critic, reward_fn, render_fn,
                                       plot_every, checkpoint_every,
                                       save_dir, use_cuda)

    for epoch in range(100):

        reward, loss = solver.train(train_loader, actor_lr, critic_lr, max_grad_norm)

        mean_valid = solver.eval(valid_loader)
        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f' % (loss, reward, mean_valid))


def train_vrp():

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset

    # VRP10, Capacity 20:  4.65  (BS) - 4.80  (Greedy)
    # VRP20, Capacity 30:  6.34  (BS) - 6.51  (Greedy)
    # VRP50, Capacity 40:  11.08 (BS) - 11.32 (Greedy)
    # VRP100, Capacity 50: 16.86 (BS) - 17.12 (Greedy)
    num_nodes = 50
    max_demand = 9
    max_load = 40
    batch_size = 128
    save_dir = 'vrp_outputs/%d_%d_%d' % (num_nodes, max_demand, max_load)

    max_grad_norm = 2.
    actor_lr = 1e-3
    critic_lr = actor_lr
    train_size = 100000
    val_size = 1000
    static_size = 2
    dynamic_size = 2
    hidden_size = 128
    dropout = 0.2
    num_layers = 1
    num_process_iter = 2

    plot_every = 500
    checkpoint_every = 250
    use_cuda = torch.cuda.is_available()

    train_data = VehicleRoutingDataset(train_size, num_nodes, max_load, max_demand)
    valid_data = VehicleRoutingDataset(val_size, num_nodes, max_load, max_demand)

    mask_fn = train_data.update_mask
    update_fn = train_data.update_dynamic
    reward_fn = vrp.reward
    render_fn = vrp.render

    actor = DRL4VRP(static_size, dynamic_size, hidden_size, update_fn, mask_fn,
                    dropout, num_layers, use_cuda)

    critic = Critic(static_size, dynamic_size, hidden_size, num_process_iter, use_cuda)

    if use_cuda:
        actor.cuda()
        critic.cuda()

    solver = NeuralCombinatorialSolver(actor, critic, reward_fn, render_fn,
                                       plot_every, checkpoint_every,
                                       save_dir, use_cuda)

    for epoch in range(100):

        train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
        valid_loader = DataLoader(valid_data, 1, True, num_workers=0)

        reward, loss = solver.train(train_loader, actor_lr, critic_lr, max_grad_norm)

        mean_valid = solver.eval(valid_loader)
        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f' % (loss, reward, mean_valid))
        print('')


if __name__ == '__main__':
    train_vrp()
    # train_tsp()
