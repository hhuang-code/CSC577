from tools.data_loader import load_data
from networks.dpp_lstm import DPPLSTM
from networks.loss import LOSS
from torch.autograd import Variable
import numpy as np
import torch

import pdb

if __name__ == '__main__':

    # load data
    data_dir = '../data/'
    test_data_name = 'SumMe'
    model_type = 2  # 1 for vsLSTM, 2 for dppLSTM
    print('...loading data')
    train_set, valid_set, test_set, test_idx = load_data(data_dir, test_data_name, model_type)

    one_sample = train_set[0][0]
    one_bin_label = train_set[1][0] # binary value: 0 - non keyframe, 1 - keyfram
    one_idx_label = train_set[2][0] # keyframe index

    seq_len = one_sample.shape[0]
    input_size = 1024
    hidden_size = 256
    output_size = 256
    c_mlp_output_size = 1
    k_mlp_output_size = 256
    num_layers = 1
    batch_size = 1

    dpplstm = DPPLSTM(input_size, hidden_size, c_mlp_output_size, 
                    k_mlp_output_size, num_layers, batch_size)

    dpp_weight = 1.0
    criterion = LOSS(dpp_weight)

    pdb.set_trace()

    one_sample = Variable(torch.from_numpy(np.expand_dims(one_sample, axis = 1)))

    c_out, k_out, L = dpplstm(one_sample)
    loss = criterion(c_out, one_bin_label, one_idx_label, L)

    print(c_out.data.shape, k_out.data.shape) 
