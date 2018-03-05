# File from: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/blob/master/neural_combinatorial_rl.py

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
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

        # Calculate the output of the decoder through an embedding + GRU
        last_embedding = self.embedding(last_output).unsqueeze(0)

        rnn_output, hidden = self.gru(last_embedding, last_hidden)
        rnn_output = rnn_output.squeeze(0)

        # Attention is applied across both the static and dynamic states of the
        # inputs, and uses the representation from the current decoders output
        attn_weights = self.attn(static_enc, dynamic_enc, rnn_output).unsqueeze(1)

        # The context vector is a weighted combination of the attention + embeddings
        context = attn_weights.bmm(static_enc.permute(0, 2, 1))

        # TODO: Convert this to use BMM instead of one-by-one
        # Using the context vector, calculate the probability of the next output
        outputs = []
        for i in range(batch_size):

            # (num_feats, 1) -> (num_feats, seq_len)
            context_ = context[i].transpose(1, 0).expand(-1, seq_len)

            # energy = (num_static_feats + num_context_feats, seq_len)
            energy = torch.cat((static_enc[i], context_), dim=0)
            energy = torch.mm(self.v, F.tanh(torch.mm(self.W, energy)))
            outputs.append(energy)

        outputs = torch.cat(outputs, dim=0)
        return outputs, hidden


class DRL4VRP(nn.Module):

    def __init__(self, static_size, dynamic_size, hidden_size, dropout,
                 num_layers, critic_beta, max_grad_norm, actor_lr,
                 actor_decay_step, actor_decay_rate, use_cuda):
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
        self.use_cuda = use_cuda

        # Define the encoder & decoder models
        self.encoder = Encoder(static_size, dynamic_size, hidden_size)
        self.decoder = Decoder(static_size, hidden_size, dropout, num_layers, use_cuda)
        self.optimizer = optim.Adam(self.parameters(), lr=self.actor_lr)

    def train(self, task, train_size, val_size, batch_size):
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

        # TODO: Change this so we pass a reward_fn to init(), and train,
        # valid DATA to the drl4vrp.train() problem
        # TODO: Force gen_dataset to yield (static, dynamic) elements from dataset
        _, reward_fn, train, valid = gen_dataset(task,  train_size, val_size)

        train_loader = DataLoader(train, batch_size, True, num_workers=0)
        valid_loader = DataLoader(valid, 1, False, num_workers=0)

        critic_ema = torch.zeros(1)
        if self.use_cuda:
            critic_ema = critic_ema.cuda()

        losses, rewards = [], []
        for batch_idx, batch in enumerate(train_loader):

            static = Variable(batch)
            dynamic = Variable(torch.zeros(static.size(0), 1, static.size(2)))
            decoder_mask = Variable(torch.ones(static.size(0), static.size(2)))

            last_output = None
            last_hidden = None
            decoder_outputs = []
            decoder_probs = []

            indices = []

            static_enc, dynamic_enc = self.encoder(static, dynamic)

            # For problems such as TSP and VRP, we choose the next "node" to visit
            # based on some probability. Given this choice, modify the dynamic
            # variables of the chosen node based on the problem
            for j in range(batch.size(2)):

                probs, last_hidden = self.decoder(static_enc, dynamic_enc,
                                                  last_output, last_hidden)

                # For TSP problems where we only visit a city once, we can speed
                # up training by preventing it from being selected again
                mask = decoder_mask.clone()

                # mask.log() will give visited cities a visit probability of -inf,
                # Idea from: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                probs = F.softmax(probs + mask.log(), dim=1)
                topi = probs.multinomial(1).data  # select an action

                # Need to select all static values to use as next decoder output,
                # so we broadcast the top indices to match the batch size
                next_output = torch.gather(batch, 2, topi.expand(-1, 2).unsqueeze(2))

                last_output = Variable(next_output.squeeze(2))
                if self.use_cuda:
                    last_output = last_output.cuda()

                # Save the data for current timestep
                probs = probs[np.arange(batch.size(0)), topi.squeeze()]

                decoder_probs.append(probs.unsqueeze(1))
                decoder_outputs.append(next_output)

                # TODO: UPDATE DYNAMIC ELEMENTS ONLY FOR THOSE THAT HAVE CHANGED
                decoder_mask[np.arange(batch.size(0)), topi.squeeze()] = 0
                # dynamic = Variable(decoder_mask.data.clone()).unsqueeze(1)
                # static_enc, dynamic_enc = self.encoder(static, dynamic)

            # Order of nodes to visit, size=(batch_size, num_feats, seq_len)
            decoder_outputs = torch.cat(decoder_outputs, dim=2)

            reward = reward_fn(decoder_outputs, self.use_cuda).sum(1)

            if batch_idx == 0:
                critic_ema = reward.mean()
            else:
                beta = self.critic_beta
                critic_ema = (critic_ema * beta) + (1. - beta) * reward.mean()

            advantage = reward - critic_ema

            # Sum the log probabilities for each city in the tour
            decoder_probs = torch.cat(decoder_probs, dim=1)
            logp_tour = torch.log(decoder_probs).sum(1)

            actor_loss = torch.mean(advantage * logp_tour)

            # Check if weights are being updated
            # a = list(p.data.numpy().copy() for p in self.parameters())

            self.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm, norm_type=2)
            self.optimizer.step()

            critic_ema = critic_ema.detach()

            # b = list(p.data.numpy().copy() for p in self.parameters())
            # print('Equal?: ', [np.allclose(a_, b_) for a_, b_ in zip(a, b)])

            # for param in self.parameters():
            #    print('grad: ', param.grad.data.sum())

            # GOAL: Average_reward for TSP20 = 3.89
            rewards.append(reward.mean().data.numpy())
            losses.append(actor_loss.data.numpy())
            lastx = 10
            if (batch_idx + 1) % lastx == 0:
                print('%d/%d, avg. reward: %2.4f, loss: %2.4f' %
                      (batch_idx, len(train_loader), np.mean(rewards[-lastx:]), np.mean(losses[-lastx:])))

        return np.mean(losses)


# "After visiting customer node i, the demands and vehicle load are updated as:
# d_{t+1}^i = max(0, d_t^i - l_t)
# d_{t+1}^k = d_t^k     for k != i
# l_{t+1} = max(0, l_t - d_t^i)
if __name__ == '__main__':

    from utils import gen_dataset
    from torch.utils.data import DataLoader

    task = 'tsp_20'
    train_size = 100000
    val_size = 1000
    batch_size = 64
    static_size = 2
    dynamic_size = 1
    hidden_size = 128
    dropout = 0.3
    use_cuda = False
    num_layers = 1
    critic_beta = 0.9
    max_grad_norm = 2.
    actor_lr = 1e-3
    actor_decay_step = 5000
    actor_decay_rate = 0.96

    '''
    a = Variable(torch.FloatTensor([[5]]), requires_grad=True)
    b = Variable(torch.FloatTensor([[1]]), requires_grad=True)
    print(id(a), id(b), id(a + b))
    import sys
    sys.exit(1)
    '''

    model = DRL4VRP(static_size, dynamic_size, hidden_size, dropout,
                    num_layers, critic_beta, max_grad_norm,
                    actor_lr, actor_decay_step, actor_decay_rate, use_cuda)

    for epoch in range(100):
        model.train(task, train_size, val_size, batch_size)
