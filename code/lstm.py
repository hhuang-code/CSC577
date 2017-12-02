import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb

# (seq_len, batch, input_feature_size)
inputs = [autograd.Variable(torch.randn(300, 1024)) for _ in range(25)]

# (input_feature_size, hidden_size, num_layers, ...)
lstm = nn.LSTM(1024, 256, 1)

# (num_layers * num_directions, batch, hidden_size)
h0 = autograd.Variable(torch.rand(1, 1, 256))

# (num_layers * num_directions, batch, hidden_size)
c0 = autograd.Variable(torch.rand(1, 1, 256))

# out: (seq_len, batch, hidden_size * num_directions)
# hidden: a tuple of (h_n, c_n)
for i in inputs:
    out, hidden = lstm(i.view(300, 1, 1024), (h0, c0))

pdb.set_trace()

x = 5
