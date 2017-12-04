import argparse
from pathlib import Path

"""
set configuration arguments as class attributes
"""
class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

"""
get configuration arguments
"""
def get_config(**kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = 'train')

    # LstmGan model args
    parser.add_argument('--input_size', type = int, default = 2048)
    parser.add_argument('--hidden_size', type = int, default = 2048)
    parser.add_argument('--num_layers', type = int, default = 2)
    parser.add_argument('--summary_rate', type = int, default = 0.2)
    
    # train
    parser.add_argument('--n_epochs', type = int, default = 1)
    parser.add_argument('--sum_learning_rate', type = float, default = 1e-4)
    parser.add_argument('--dis_learning_rate', type = float, default = 1e-5)
    parser.add_argument('--dis_start_batch', type = int, default = 5)
    
    # log
    parser.add_argument('--log_dir', type = str, default = Path('/tmp/log/'))
    parser.add_argument('--detail_flag', type = bool, default = True)
    
    args = parser.parse_args()

    # namespace -> dictionary
    args = vars(args)
    args.update(kwargs)

    return Config(**args)
