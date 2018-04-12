import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):

        output = self.conv(input)  # (batch, in, seq_len) -> (batch, out, seq_len)
        return output


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static, dynamic and decoder elements
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.W = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))

    def forward(self, static_enc, dynamic_enc, decoder_hidden):

        batch_size, hidden_size, _ = static_enc.size()

        # Energy shape is: (batch, total_num_feats, seq_len)
        hidden = decoder_hidden.permute(1, 2, 0).expand_as(static_enc)
        energy = torch.cat((static_enc, dynamic_enc, hidden), dim=1)

        v_view = self.v.unsqueeze(0).expand(batch_size, 1, hidden_size)
        W_view = self.W.unsqueeze(0).expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v_view, F.tanh(torch.bmm(W_view, energy)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns


class Decoder(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, output_size, hidden_size, dropout=0.2, num_layers=1):
        super(Decoder, self).__init__()

        # Use a learnable initial state (x0) if none is provided
        self.x0 = nn.Parameter(torch.FloatTensor(1, output_size))
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.W = nn.Parameter(torch.FloatTensor(hidden_size, 2 * hidden_size))

        # Used to compute a representation of the current decoder output
        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.attn = Attention(hidden_size)

    def forward(self, static_enc, dynamic_enc, last_output, last_hidden):

        batch_size, _, _ = static_enc.size()

        # If we're solving e.g. TSP with no initial state - learn one instead
        if last_output is None:
            last_output = self.x0.expand(batch_size, -1)

        last_embedding = self.embedding(last_output).unsqueeze(0)
        rnn_out, hidden = self.gru(last_embedding, last_hidden)

        # Attention is applied across the static and dynamic states of the input
        attn = self.attn(static_enc, dynamic_enc, rnn_out)  # (B, 1, seq_len)

        # The context vector is a weighted combination of the attention + inputs
        context = attn.bmm(static_enc.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.squeeze(1).unsqueeze(2).expand_as(static_enc)
        energy = torch.cat((static_enc, context), dim=1)  # (B, num_feats, seq_len)

        W_view = self.W.unsqueeze(0).expand(batch_size, -1, -1)
        v_view = self.v.unsqueeze(0).expand(batch_size, -1, -1)

        probs = torch.bmm(v_view, F.tanh(torch.bmm(W_view, energy)))
        probs = probs.squeeze(1)

        return probs, hidden


class DRL4TSP(nn.Module):
    """Defines the main Encoder + Decoder combinatorial model.

    The TSP is defined by the following traits:
        1. Each city in the list must be visited once and only once, which sets
           an upper bound on the number of steps to be performed
        2. The salesman must return to the original node at the end of the tour

    Variants on this scheme can be introduced, such as:
        1. Only traveling a subset of the path
        2. Giving dynamic variables to the cities. By default the city generator
           assumes a dynamic vector composed of 0's, and we do some slightly 
           inefficient computations in this case. Improvements could be done to 
           only use static elements.

    Parameters
    ----------
    mask_fn:
        Defined by the task, and used to speed up learning by preventing certain
        states from being selected by masking with a probability of -inf.

        See: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5

    """

    def __init__(self, static_size, dynamic_size, hidden_size, update_fn=None,
                 mask_fn=None, dropout=0., num_layers=1, use_cuda=False):
        super(DRL4TSP, self).__init__()

        self.update_fn = update_fn
        self.mask_fn = mask_fn
        self.use_cuda = use_cuda

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Decoder(static_size, hidden_size, dropout, num_layers)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform(p)

    def forward(self, static, dynamic, last_output=None, last_hidden=None):

        # Structures for holding the output sequences
        tour_idx = []
        tour_logp = []

        if self.use_cuda:
            mask = torch.cuda.FloatTensor(static.size(0), static.size(2)).fill_(1)
        else:
            mask = torch.FloatTensor(static.size(0), static.size(2)).fill_(1)

        # Begin optimization - static is only ever processed once, while dynamic
        # may be updated on each iteration, depending on the problem
        static_enc = self.static_encoder(static)
        dynamic_enc = self.dynamic_encoder(dynamic)

        step = 0
        while step < static.size(2) and mask.byte().any():
            step = step + 1

            probs, last_hidden = self.decoder(static_enc, dynamic_enc,
                                              last_output, last_hidden)

            mask_var = Variable(mask, requires_grad=False)
            probs = F.softmax(probs + mask_var.log(), dim=1)

            if self.training:
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()

                # Sometimes an issue with Categorical & sampling on GPU;
                # ensure we're only sampling from indices defined by the mask
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            view = ptr.view(-1, 1, 1).expand(-1, static.size(1), -1)
            last_output = torch.gather(static, 2, view).squeeze(2)

            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_enc = self.dynamic_encoder(dynamic)

            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data)

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1).sum(1)  # (batch_size,)
        return tour_idx, tour_logp


class DRL4VRP(nn.Module):
    """Defines the main Encoder + Decoder combinatorial model.

    This module differs from the TSP module in the following manner:
    1. Each sequence can get processed a different number of times
    2. Each sequence that changes gets stored individually in a list of lists,
       making the approach less efficient.
    """

    def __init__(self, static_size, dynamic_size, hidden_size, update_fn=None,
                 mask_fn=None, dropout=0., num_layers=1, use_cuda=False):
        super(DRL4VRP, self).__init__()

        self.update_fn = update_fn
        self.mask_fn = mask_fn
        self.use_cuda = use_cuda

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Decoder(static_size, hidden_size, dropout, num_layers)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform(p)

    def forward(self, static, dynamic, last_output=None, last_hidden=None):

        # Structures for holding the output sequences
        tour_idx = []
        tour_logp = [[] for _ in range(static.size(0))]

        if self.use_cuda:
            mask = torch.cuda.FloatTensor(static.size(0), static.size(2)).fill_(1)
        else:
            mask = torch.FloatTensor(static.size(0), static.size(2)).fill_(1)

        # Begin optimization - static is onsly ever processed once, while dynamic
        # may be updated on each iteration, depending on the problem
        static_enc = self.static_encoder(static)
        dynamic_enc = self.dynamic_encoder(dynamic)

        step = 0
        max_step = static.size(2) if self.mask_fn is None else 1000
        while step < max_step and mask.byte().any():
            step = step + 1

            probs, last_hidden = self.decoder(static_enc, dynamic_enc,
                                              last_output, last_hidden)

            probs = F.softmax(probs + Variable(mask).log(), dim=1)

            if self.training:
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()

                # Sometimes an issue with Categorical & sampling on GPU;
                # ensure we're only sampling from indices defined by the mask
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp_ = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)
                logp_ = prob.log()

            # Only save the tour probs if the remaining demand in tour is > 0
            is_active_mask = dynamic.data[:, 1, 1:].sum(1).gt(0).float()
            for idx in is_active_mask.nonzero().squeeze(1):
                tour_logp[idx].append(logp_[idx])

            tour_idx.append(ptr.data.unsqueeze(1))

            view = ptr.view(-1, 1, 1).expand(-1, static.size(1), -1)
            last_output = torch.gather(static, 2, view).squeeze(2)

            # Update the dynamics variables
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_enc = self.dynamic_encoder(dynamic)

            # Update the mask
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data)

        # (batch_size, seq_len)
        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat([torch.cat(p_).sum() for p_ in tour_logp])

        return tour_idx, tour_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
