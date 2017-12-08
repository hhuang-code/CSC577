from pathlib import Path

from torchvision.datasets.folder import default_loader
from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable

import h5py
import numpy as np
from tqdm import tqdm

from config import get_config
from tools.fea_extractor import *
from tools.data_loader import feature_loader
from networks.Summarizer import Summarizer
from LstmGan import LstmGan

import pdb

def pre_process(config):
    # get frames from video
    get_frames(config.video_dir_youtube, config.frame_dir_youtube)
    
    # ger resnet feature for each video
    get_res_feature(config.frame_dir_youtube, config.feature_dir_youtube)

if __name__ == '__main__':

    # generate ResNet feature
    #get_res_feature()
    
    config = get_config(mode = 'train')
    
    #pre_process(config)

    feature_dir = config.feature_dir_youtube
    
    train_loader = feature_loader(feature_dir, 'train')

    #test_loader = feature_loader(feature_dir, 'test')
    test_loader = None

    lstmgan = LstmGan(config, train_loader, test_loader)

    lstmgan.build()

    lstmgan.train()
