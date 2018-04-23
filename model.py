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


class DRL4VRP(nn.Module):
    """Defines the main Encoder + Decoder combinatorial model.


    Variants on this scheme can be introduced, such as:
        1. Only traveling a subset of the path
        2. Giving dynamic variables to the cities. By default the city generator
           assumes a dynamic vector composed of 0's, and we do some slightly
           inefficient computations in this case. Improvements could be done to
           only use static elements.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask 
        is provided, we terminate the search after a fixed number of iterations 
        to avoid tours that stretch forever
    dropout: float
        Defines the dropout rate for the decoder
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    use_cuda: bool
        Use the GPU or not
    """

    def __init__(self, static_size, dynamic_size, hidden_size, update_fn=None,
                 mask_fn=None, dropout=0., num_layers=1, use_cuda=False):
        super(DRL4VRP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

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
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        last_output: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hidden: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
    "   """
        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []

        if self.use_cuda:
            mask = torch.cuda.FloatTensor(static.size(0), static.size(2)).fill_(1)
        else:
            mask = torch.FloatTensor(static.size(0), static.size(2)).fill_(1)

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        # An improvement could be to only process those elements that recently
        # changed, and not iterate over the full input space again.
        static_enc = self.static_encoder(static)
        dynamic_enc = self.dynamic_encoder(dynamic)

        step = 0
        max_step = static.size(2) if self.mask_fn is None else 1000
        while step < max_step and mask.byte().any():

            probs, last_hidden = self.decoder(static_enc, dynamic_enc,
                                              last_output, last_hidden)

            mask_var = Variable(mask, requires_grad=False)
            probs = F.softmax(probs + mask_var.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()

                # Sometimes an issue with Categorical & sampling on GPU;
                # ensure we're only sampling from indices defined by the mask
                # See: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
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

                #dyn = torch.gather(dynamic, 2, view)
                #decoded = self.dynamic_encoder(dyn)
                #dynamic_enc.scatter(2, view, decoded)

            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data)

            step = step + 1

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1).sum(1)  # (batch_size,)
        return tour_idx, tour_logp


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
