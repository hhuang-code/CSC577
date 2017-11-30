import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb

# dpp_lstm model
class DPPLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bias = True):
        super(DPPLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_lyaers
        self.output_size = output_size
    
        # forward lstm
        self.forward_lstm = LSTM(input_size, hidden_size, 
                                num_layers = num_layers, bias = bias)
        # backward lstm
        self.back_lstm = LSTM(input_size, hidden_size,
                                num_layers = num_layers, bias = bias)
        
        # classify mlp
        self.classify_mlp = nn.Linear(input_size + 2 * output_size, output_size, bias = bias)

        # kernel mlp
        self.kernel_mlp = nn.Linear(input_size + 2 * output_size, output_size, bias = bias)

    def forward(self, x):
                    

# (seq_len, batch, input_feature_size)
inputs = [autograd.Variable(torch.randn(300, 1024)) for _ in range(2)]

# (input_feature_size, hidden_size, num_layers, ...)
lstm = nn.LSTM(1024, 256, 1)

# (num_layers * num_directions, batch, hidden_size)
h0 = autograd.Variable(torch.rand(1, 1, 256))

# (num_layers * num_directions, batch, hidden_size)
c0 = autograd.Variable(torch.rand(1, 1, 256))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), (h0, c0))
    pdb.set_trace()
    y = 1

x = 5
