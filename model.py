# File from: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/blob/master/neural_combinatorial_rl.py

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


class Encoder(nn.Module):

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(Encoder, self).__init__()

        self.static_size = static_size
        self.dynamic_size = dynamic_size
        self.hidden_size = hidden_size

        # 5.2: "Each customer location is also embedded into a vector of size
        # 128, shared among the inputs"
        # Conv1d: input=(N, *, in_features), output=(N, *, out_features)
        # See: http://pytorch.org/docs/master/nn.html#torch.nn.Conv1d
        self.sconv = nn.Conv1d(static_size, hidden_size, kernel_size=1, bias=False)

        # 5.2: "We employ similar embeddings for the dynamic elements; the demand
        # d_t^i and the remaining vehicle load after visiting node i, l_t - d_t^i
        # are mapped to a vector in a 128-dimensional vector space and used in
        # the attention layer
        self.dconv = nn.Conv1d(dynamic_size, hidden_size, kernel_size=1, bias=False)

    def forward(self, static_in, dynamic_in):

        static = self.sconv(static_in)
        dynamic = self.dconv(dynamic_in)
        return (static, dynamic)


class Attention(nn.Module):

    def __init__(self, hidden_size, use_cuda=False):
        super(Attention, self).__init__()

        # The static, dynamic, and hidden (from decoder) all have the same size
        if use_cuda:
            self.v = nn.Parameter(torch.FloatTensor(1, 3 * hidden_size).cuda())
        else:
            self.v = nn.Parameter(torch.FloatTensor(1, 3 * hidden_size))

        if use_cuda:
            self.W = nn.Parameter(torch.FloatTensor(3 * hidden_size, 3 * hidden_size).cuda())
        else:
            self.W = nn.Parameter(torch.FloatTensor(3 * hidden_size, 3 * hidden_size))
        self.use_cuda = use_cuda

    def forward(self, encoder_outputs, decoder_hidden):

        # Static / dynamic = (batch, num_nodes, num_feats)
        static, dynamic = encoder_outputs

        # Create variable to store attention energies (batch_size, num_nodes)
        attns = Variable(torch.zeros(static.size(0), static.size(1)))

        if self.use_cuda:
            attns = attns.cuda()

        for i in range(static.size(0)):
            for j in range(static.size(1)):

                cat = torch.cat((static[i, j], dynamic[i, j], decoder_hidden[i]))
                cat = cat.unsqueeze(1)

                print('static/dynamic: ', static.shape, dynamic.shape)
                print('shapes: ', self.v.shape, self.W.shape, cat.shape)
                attns[i, j] = self.v.dot(F.tanh(self.W.dot(cat)))

        attns = F.softmax(attns, dim=1)
        return attns


class Decoder(nn.Module):

    def __init__(self, output_size, hidden_size, dropout=0.2, num_layers=1, use_cuda=False):
        super(Decoder, self).__init__()

        self.use_cuda = use_cuda

        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)

        # Concatenating the static & attention params, which have equal size
        size = 2 * hidden_size

        if use_cuda:
            self.x0 = nn.Parameter(torch.FloatTensor(1, output_size).cuda())
            self.h0 = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size).cuda())
            self.v = nn.Parameter(torch.FloatTensor(1, size).cuda())
            self.W = nn.Parameter(torch.FloatTensor(size, size).cuda())
        else:
            self.x0 = nn.Parameter(torch.FloatTensor(1, output_size))
            self.h0 = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
            self.v = nn.Parameter(torch.FloatTensor(1, size))
            self.W = nn.Parameter(torch.FloatTensor(size, size))

        self.attn = Attention(hidden_size, use_cuda)

    def forward(self, decoder_input, last_hidden, encoder_outputs):

        # Static / dynamic = (batch, num_nodes, num_feats)
        static, dynamic = encoder_outputs

        # Holds probabilities for each state being next (batch_size, num_nodes)
        outputs = Variable(torch.zeros(static.size(0), static.size(1)))
        if self.use_cuda:
            outputs = outputs.cuda()

        # Use a learnable hidden state & input for the decoder
        if last_hidden is None:
            last_hidden = self.h0
        if decoder_input is None:
            decoder_input = self.x0.repeat(static.size(0), 1)

        # decoder_input is the static state (i.e. lat / long only)
        # To interface with the RNN, treat each "step" as a sequence of length 1
        embedded = self.embedding(decoder_input).unsqueeze(0)
        last_hidden = last_hidden.unsqueeze(0)

        # Calculate the new hidden state using embedding & previous state
        # See: http://pytorch.org/docs/master/nn.html#torch.nn.GRU
        # input=(seq_len, batch, input_size)
        # h_0=(num_layers * num_directions, batch, hidden_size)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        rnn_output = rnn_output.squeeze(0)

        # Attention is applied across both the static and dynamic states of the
        # inputs, and uses the representation from the current decoders output
        attn_weights = self.attn(encoder_outputs, rnn_output)

        # attn_weights = (batch, 1, num_nodes), static=(batch, num_nodes, num_feats)
        context = attn_weights.bmm(static.transpose(0, 1))
        context = context.squeeze(1)  # (batch, num_feats)

        # Calculate the probability of visiting each possible node
        for i in range(static.size(0)):  # sample i
            cat = torch.cat((static[i], contact), dim=1)
            outputs[i] = self.v.dot(F.tanh(self.W.dot(cat)))

        outputs = F.softmax(outputs, dim=1)

        return outputs, hidden


class DRL4VRP(nn.Module):

    def __init__(self, task, static_size, dynamic_size, hidden_size, dropout,
                 num_layers, use_cuda):
        super(DRL4VRP, self).__init__()

        self.task = task
        self.static_size = static_size
        self.dynamic_size = dynamic_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_cuda = use_cuda

        self.encoder = Encoder(static_size, dynamic_size, hidden_size)
        self.decoder = Decoder(static_size, hidden_size, dropout, num_layers, use_cuda)

    def train(self, train_size, val_size, batch_size):
        """
        Procedure
        ---------
        1. Given all possible input nodes, compute:
            a. An embedding for the static state
            b. An embedding for the dynamic state
        2. Calculate a hidden representation & output from last item in sequence
        3. Calculate attention using (embeddings, output_rep)
        4. Calculate new output using argmax
        5. Update data for the dynamic element that was chosen, and compute a
            new dynamic embedding
        """

        input_dim, reward_fn, train, valid = gen_dataset(self.task,  train_size, val_size)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=1)

        decoder_input = None
        last_hidden = None
        for batch in train_loader:

            static = Variable(batch)
            dynamic = Variable(torch.FloatTensor(static.size(0), 1, static.size(2)).fill_(1))

            encoder_outputs = self.encoder(static, dynamic)

            for node in range(batch.size(2)):

                decoder_probs, last_hidden = self.decoder(
                    decoder_input, last_hidden, encoder_outputs)

                # The decoder output is the input node with highest probability
                topv, topi = decoder_probs.data.topk(1)

                decoder_input = Variable(torch.FloatTensor([[topi[0][0]]]))
                if self.use_cuda:
                    decoder_input = decoder_input.cuda()

                # UPDATE DYNAMIC ELEMENTS
                # CREATE MASK TO HIDE ELEMENT FROM BEING SELECTED (SPEED UP TRAINING)


# "After visiting customer node i, the demands and vehicle load are updated as:
# d_{t+1}^i = max(0, d_t^i - l_t)
# d_{t+1}^k = d_t^k     for k != i
# l_{t+1} = max(0, l_t - d_t^i)
if __name__ == '__main__':

    from trainer import gen_dataset
    from torch.utils.data import DataLoader

    task = 'tsp_10'
    train_size = 1000
    val_size = 1000
    batch_size = 128
    static_size = 2
    dynamic_size = 1
    hidden_size = 128
    dropout = 0.5
    use_cuda = False
    num_layers = 1

    model = DRL4VRP(task, static_size, dynamic_size, hidden_size, dropout,
                    num_layers, use_cuda)
    model.train(train_size, val_size, batch_size)
