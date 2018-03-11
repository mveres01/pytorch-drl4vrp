# File from: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/blob/master/neural_combinatorial_rl.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import make_dot


class Encoder(nn.Module):

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(Encoder, self).__init__()

        # Use a 1d CNN to embed both the static & dynamic elements
        self.sconv = nn.Conv1d(static_size, hidden_size, kernel_size=1)
        self.dconv = nn.Conv1d(dynamic_size, hidden_size, kernel_size=1)

    def forward(self, static_in, dynamic_in):

        # (batch_size, input_features, seq_len) -> (batch_size, hidden_size, seq_len)
        static = self.sconv(static_in)
        dynamic = self.dconv(dynamic_in)

        return static, dynamic


class Attention(nn.Module):

    def __init__(self, hidden_size, use_cuda=False):
        super(Attention, self).__init__()

        # The static, dynamic, and hidden (from decoder) all have the same size
        if use_cuda:
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size).cuda())
            self.W = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size).cuda())
        else:
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
            self.W = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))

        for p in self.parameters():
            nn.init.xavier_uniform(p)

        self.use_cuda = use_cuda

    def forward(self, static_enc, dynamic_enc, decoder_hidden):

        batch_size, _, seq_len = static_enc.size()

        # Attention is calculated across the input sequence
        if self.use_cuda:
            attns = Variable(torch.zeros(batch_size, seq_len).cuda())
        else:
            attns = Variable(torch.zeros(batch_size, seq_len))

        # TODO: Chage this to BMM instead of doing one at a time
        for i in range(batch_size):

            # Expand the hidden rep so we can use a single matrix-multiply to
            # calculate energy across all nodes in sequence (num_feats, seq_len)
            hidden = decoder_hidden[i].unsqueeze(1).expand(-1, seq_len)

            energy = torch.cat((static_enc[i], dynamic_enc[i], hidden), 0)

            attns[i] = torch.mm(self.v, F.tanh(torch.mm(self.W, energy)))

        attns = F.softmax(attns, dim=1)  # (batch, seq_len)
        return attns


class Decoder(nn.Module):

    def __init__(self, output_size, hidden_size, dropout=0.2, num_layers=1, use_cuda=False):
        super(Decoder, self).__init__()

        # Use a learnable initial state (x0) & hidden representation (h0), with
        # v & W used to compute the output with attentions
        if use_cuda:
            self.x0 = nn.Parameter(torch.FloatTensor(1, output_size).cuda())
            self.h0 = nn.Parameter(torch.FloatTensor(1, hidden_size).cuda())
            self.v = nn.Parameter(torch.FloatTensor(1,  hidden_size).cuda())
            self.W = nn.Parameter(torch.FloatTensor(hidden_size, 2 * hidden_size).cuda())
        else:
            self.x0 = nn.Parameter(torch.FloatTensor(1, output_size))
            self.h0 = nn.Parameter(torch.FloatTensor(1, hidden_size))
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
            self.W = nn.Parameter(torch.FloatTensor(hidden_size, 2 * hidden_size))

        for p in self.parameters():
            nn.init.xavier_uniform(p)

        self.use_cuda = use_cuda

        # Used to compute a representation of the current decoder output
        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.attn = Attention(hidden_size, use_cuda)

    def forward(self, static_enc, dynamic_enc, last_output, last_hidden):

        batch_size, _, seq_len = static_enc.size()

        # Use a learnable hidden state & input for the decoder
        if last_hidden is None:
            last_hidden = self.h0.unsqueeze(0).expand(-1, batch_size, -1)
        if last_output is None:
            last_output = self.x0.expand(batch_size, -1)  # (batch_size, input_size)

        last_embedding = self.embedding(last_output).unsqueeze(0)
        rnn_output, hidden = self.gru(last_embedding, last_hidden)
        rnn_output = rnn_output.squeeze(0)

        # Attention is applied across the static and dynamic states of the input
        attn_weights = self.attn(static_enc, dynamic_enc, rnn_output).unsqueeze(1)

        # The context vector is a weighted combination of the attention + inputs
        context = attn_weights.bmm(static_enc.permute(0, 2, 1))

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.permute(0, 2, 1).expand(-1, -1, seq_len)
        context = torch.cat((static_enc, context), dim=1)
        W_view = self.W.unsqueeze(0).expand(batch_size, -1, -1)
        v_view = self.v.unsqueeze(0).expand(batch_size, -1, -1)

        outputs = torch.bmm(v_view, F.tanh(torch.bmm(W_view, context)))
        outputs = outputs.squeeze(1)

        return outputs, hidden


class DRL4VRP(nn.Module):

    def __init__(self, static_size, dynamic_size, hidden_size, dropout,
                 num_layers, critic_beta, max_grad_norm, actor_lr,
                 actor_decay_step, actor_decay_rate, plot_every, use_cuda):
        super(DRL4VRP, self).__init__()

        self.static_size = static_size
        self.dynamic_size = dynamic_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.critic_beta = critic_beta
        self.max_grad_norm = max_grad_norm
        self.actor_lr = actor_lr
        self.actor_decay_step = actor_decay_step
        self.actor_decay_rate = actor_decay_rate
        self.plot_every = plot_every
        self.use_cuda = use_cuda

        # Define the encoder & decoder models
        self.encoder = Encoder(static_size, dynamic_size, hidden_size)
        self.decoder = Decoder(static_size, hidden_size, dropout, num_layers, use_cuda)
        self.optimizer = optim.Adam(self.parameters(), lr=self.actor_lr)

    def forward(self, static, dynamic, last_output=None, update_fn=None, mask_fn=None):
        """
        Parameters
        ----------
        mask_fn:
            We can speed up learning by preventing states from being selected
            by forcing a prob of -inf.
        """

        last_hidden = None
        decoder_probs = []
        decoder_indices = []
        decoder_mask = Variable(torch.ones(static.size(0), static.size(2)))

        static_enc, dynamic_enc = self.encoder(static, dynamic)

        # If we've supplied a masking function, we'll use that to determine when
        # to stop (and use an arbitrary number of iters). Otherwise, treat the
        # problem as a TSP, and only perform iterations equal to the number of nodes
        max_iters = static.size(2) if mask_fn is None else 1000

        for _ in range(max_iters):

            probs, last_hidden = self.decoder(static_enc, dynamic_enc,
                                              last_output, last_hidden)

            mask = decoder_mask.clone()

            # Use mask.log() to prevent certain indices from being selected.
            # Idea from: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
            probs = F.softmax(probs + mask.log(), dim=1)
            top_k = torch.distributions.Categorical(probs).sample().data

            # Broadcast top choice to collect ALL corresponding static elements
            top_k_view = top_k.view(-1, 1, 1).expand(-1, static.size(1), -1)
            next_output = torch.gather(static.data, 2, top_k_view)

            last_output = Variable(next_output.squeeze(2))
            if self.use_cuda:
                last_output = last_output.cuda()

            # Keep the probability of choosing the specific action
            probs = probs[np.arange(static.size(0)), top_k]
            decoder_probs.append(probs.unsqueeze(1))
            decoder_indices.append(top_k.unsqueeze(1))

            # Update the dynamics variables
            if update_fn is not None:
                dynamic = update_fn(dynamic.clone(), top_k)
                static_enc, dynamic_enc = self.encoder(static, dynamic)

            # Update the mask
            if mask_fn is not None:
                decoder_mask = mask_fn(decoder_mask, dynamic, top_k)
                if not decoder_mask.byte().any():
                    break

        decoder_probs = torch.cat(decoder_probs, dim=1)  # (batch_size, seq_len)
        decoder_indices = torch.cat(decoder_indices, dim=1)
        return decoder_indices, decoder_probs

    def train(self, trainset, valset, reward_fn, batch_size):
        """
        Procedure
        ---------
        1. Calculate an embedding for static & dynamic elements
        2. Calculate an representation of the last decoder output
        3. Calculate an attention vector
        4. Blend the attention and static embeddings to get a context vector
        5. Calculate a new output via context vector & static embeddings
        6. Update data for the chosen dynamic element & compute new embedding
        """

        train_loader = DataLoader(trainset, batch_size, True, num_workers=0)
        valid_loader = DataLoader(valset, 1, False, num_workers=0)

        # (Possibly) use a mask to speed up training
        mask_fn = getattr(trainset, 'update_mask', None)
        update_fn = getattr(trainset, 'update_dynamic', None)

        critic_ema = torch.zeros(1)
        if self.use_cuda:
            critic_ema = critic_ema.cuda()

        losses, rewards = [], []
        for batch_idx, batch in enumerate(train_loader):

            start = time.time()

            static = Variable(batch[0])
            dynamic = Variable(batch[1])
            initial_state = Variable(batch[2]) if len(batch[2]) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_probs = self.forward(static, dynamic,
                                                    initial_state,
                                                    update_fn, mask_fn)

            reward = reward_fn(static, tour_indices, self.use_cuda).sum(1)

            # Update the network
            if batch_idx == 0:
                critic_ema = reward.mean()
            else:
                beta = self.critic_beta
                critic_ema = (critic_ema * beta) + (1. - beta) * reward.mean()
            advantage = (reward - critic_ema)

            # Sum the log probabilities for each city in the tour
            logp_tour = torch.log(tour_probs).sum(1)
            actor_loss = torch.mean(advantage * logp_tour)

            self.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm, norm_type=2)
            self.optimizer.step()

            critic_ema = critic_ema.detach()

            # GOALS: TSP_20=3.97, TSP_50=6.08, TSP_100=8.44
            rewards.append(reward.mean().data.numpy())
            losses.append(actor_loss.data.numpy())
            if (batch_idx + 1) % self.plot_every == 0:

                if not os.path.exists('outputs'):
                    os.makedirs('outputs')
                trainset.render(static, tour_indices)
                plt.savefig('outputs/%d.png' % batch_idx)

                print('%d/%d, avg. reward: %2.4f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader),
                       np.mean(rewards[-self.plot_every:]),
                       np.mean(losses[-self.plot_every:]), time.time() - start))

        return np.mean(losses)

    def train_multilength(self, trainset, valset, reward_fn, batch_size):
        """
        Procedure
        ---------
        1. Calculate an embedding for static & dynamic elements
        2. Calculate an representation of the last decoder output
        3. Calculate an attention vector
        4. Blend the attention and static embeddings to get a context vector
        5. Calculate a new output via context vector & static embeddings
        6. Update data for the chosen dynamic element & compute new embedding
        """

        train_loader = DataLoader(trainset, batch_size, True, num_workers=0)
        valid_loader = DataLoader(valset, 1, False, num_workers=0)

        # (Possibly) use a mask to speed up training
        mask_fn = getattr(trainset, 'update_mask', None)
        update_fn = getattr(trainset, 'update_dynamic', None)

        critic_ema = torch.zeros(1)
        if self.use_cuda:
            critic_ema = critic_ema.cuda()

        losses, rewards = [], []
        for batch_idx, batch in enumerate(train_loader):

            start = time.time()

            static = Variable(batch[0])
            dynamic = Variable(batch[1])
            initial_state = Variable(batch[2]) if len(batch[2]) > 0 else None

            tour_indices, tour_probs, reward = [], [], []

            # for static_, dynamic_, state_ in zip(static, dynamic, initial_state):
            for i in range(len(static)):

                state = None if initial_state is None else initial_state[i:i + 1]
                idx, prob = self.forward(static[i:i + 1],
                                         dynamic[i:i + 1],
                                         state,
                                         update_fn, mask_fn)
                tour_indices.append(idx)
                tour_probs.append(torch.log(prob).sum())
                reward.append(reward_fn(static[i:i + 1], idx, self.use_cuda).sum(1))

            logp_tour = torch.cat(tour_probs)
            reward = torch.cat(reward)

            # Update the network
            if batch_idx == 0:
                critic_ema = reward.mean()
            else:
                beta = self.critic_beta
                critic_ema = (critic_ema * beta) + (1. - beta) * reward.mean()

            advantage = (reward - critic_ema)

            # Sum the log probabilities for each city in the tour
            actor_loss = torch.mean(advantage * logp_tour)

            self.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm, norm_type=2)
            self.optimizer.step()

            critic_ema = critic_ema.detach()

            # GOALS: TSP_20=3.97, TSP_50=6.08, TSP_100=8.44
            rewards.append(reward.mean().data.numpy())
            losses.append(actor_loss.data.numpy())
            if (batch_idx + 1) % self.plot_every == 0:

                if not os.path.exists('outputs'):
                    os.makedirs('outputs')

                trainset.render(static, tour_indices)
                plt.savefig('outputs/%d.png' % batch_idx)

                print('%d/%d, avg. reward: %2.4f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader),
                       np.mean(rewards[-self.plot_every:]),
                       np.mean(losses[-self.plot_every:]), time.time() - start))

        return np.mean(losses)


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
