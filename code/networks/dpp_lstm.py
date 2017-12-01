import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pdb

"""
dpp_lstm model
Args:
    input_size: num of input features
    hidden_size: num of hidden units of lstm
    c_mlp_output_size: num of output features of classify mlp
    k_mlp_output_size: num of output features of kernel mlp
    num_layers: num of lstm layers
"""
class DPPLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, c_mlp_output_size, k_mlp_output_size, num_layers, batch_size, bias = True):
        super(DPPLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.c_mlp_output_size = c_mlp_output_size
        self.k_mlp_output_size = k_mlp_output_size
        self.num_layers = num_layers
    
        # forward lstm
        self.forward_lstm = nn.LSTM(input_size, hidden_size, 
                                num_layers = num_layers, bias = bias)
        # backward lstm
        self.backward_lstm = nn.LSTM(input_size, hidden_size,
                                num_layers = num_layers, bias = bias)
        
        # classify mlp
        self.classify_mlp = nn.Linear(input_size + 2 * hidden_size, c_mlp_output_size, bias = bias)

        # sigmoid layer, on the top of classify_mlp
        self.sigmoid = nn.Sigmoid()

        # kernel mlp
        self.kernel_mlp = nn.Linear(input_size + 2 * hidden_size, k_mlp_output_size, bias = bias)

        # initial forward lstm hidden states
        self.forward_hidden = self.init_hidden(num_layers, batch_size, hidden_size) 

        # initial back lstm hidden states
        self.backward_hidden = self.init_hidden(num_layers, batch_size, hidden_size)

    # initialize h0 and c0, shape of (num_layers * num_directions, batch_size, hidden_size)
    def init_hidden(self, num_layers, batch_size, hidden_size):
        h0 = Variable(torch.randn(num_layers * 1, batch_size, hidden_size))
        c0 = Variable(torch.randn(num_layers * 1, batch_size, hidden_size))
        
        return (h0, c0)
    
    """
    Return:
        c_out: output of classify_mlp
        k_out: output of kernel_mlp
        L: kernel matrix, used to calculate dpp loss
    """
    def forward(self, x):
        pdb.set_trace()
        # forward pass
        forward_lstm_out, _ = self.forward_lstm(x, self.forward_hidden)

        # reverse x along seq_len axis
        seq_len = x.data.shape[0]   # num of frames
        indices = torch.linspace(seq_len - 1, 0, seq_len).long()
        r_x = torch.index_select(x, 0, indices)
        
        # backward pass
        backward_lstm_out, _ = self.backward_lstm(r_x, self.backward_hidden)
        
        # concatenate input for c_mlp and k_mlp
        mlp_in = torch.cat((forward_lstm_out, backward_lstm_out, x), 2)
        
        # classify mlp pass, sigmoid
        c_out = self.classify_mlp(mlp_in)
        c_out = self.sigmoid(c_out)

        # kernel mlp pass, linear
        k_out = self.kernel_mlp(mlp_in)

        """
        calculate kernel matrix L
        """
        pdb.set_trace()
        # reshape c_out to 1D vector, and get outer product
        c_mat = torch.ger(c_out.view(seq_len), c_out.view(seq_len))

        # reshape k_out to 2D matrix, and get inner product
        k_mat = torch.mm(k_out.view(seq_len, self.k_mlp_output_size), torch.t(k_out.view(seq_len, self.k_mlp_output_size)))

        # kernel matrix
        L = torch.mm(c_mat, k_mat)

        return (c_out, k_out, L)
