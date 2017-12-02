import torch
import torch.nn as nn
from torch.autograd import Variable

import pdb

# output normalized importance scores
class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2, bidirectional = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        pdb.set_trace()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional = bidirectional)
        self.to_score = nn.Sequential(
                nn.Linear(hidden_size * 2, 1) # bidirection -> scalar
                )
        self.hidden_0 = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_size).cuda()),
                Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_size)).cuda())

    """
    Args:
        x: ResNet features, (seq_len, 1, 2048)
    Return:
        scores: normalized, (seq_len, 1)
    """
    def forward(self, x):
        pdb.set_trace()
        # output: (seq_len, 1, hidden_size * 2)
        output, (h_n, c_n) = self.lstm(x, self.hidden_0)
        # scores: (seq_len, 1)
        scores = self.to_score(output.squeeze(1))

        return scores

# consist of sLSTM, eLSTM, dLSTM
class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.slstm = sLSTM(input_size, hidden_size, num_layers, True)

    def forward(self, x, score_weighted_flag = True):
        # input sequence fream features are weighted with scores
        if score_weighted_flag:
            scores = self.slstm(x)
            # weighted features: (seq_len, 1, 2048)
            pdb.set_trace()
            weighted_feature = x * scores.view(-1, 1, 1)    # broadcasting
        else:
            weighted_feature = x

        return weighted_feature
