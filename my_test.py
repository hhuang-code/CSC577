from pathlib import Path

from torchvision.datasets.folder import default_loader
from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable

import h5py
import numpy as np
from tqdm import tqdm

from config import get_config
from tools.fea_extractor import get_res_feature
from tools.data_loader import feature_loader
from networks.Summarizer import Summarizer
from LstmGan import LstmGan

import pdb

if __name__ == '__main__':

    # generate ResNet feature
    #get_res_feature()
    
    config = get_config(mode = 'train')

    #pdb.set_trace()
    fea_dir = Path('/home/aaron/Documents/Courses/577/dataset/feature/Youtube')
    
    train_loader = feature_loader(fea_dir, 'train')

    test_loader = feature_loader(fea_dir, 'test')

    lstmgan = LstmGan(config, train_loader, test_loader)

    lstmgan.build()

    lstmgan.train()
