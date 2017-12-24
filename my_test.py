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
from tools.data_loader import *
from networks.Summarizer import Summarizer
from LstmGan import LstmGan

import pdb

def pre_process(config):
    # get frames from video
    get_frames(config.video_dir_tvsum, config.frame_dir_youtube)
    
    # ger resnet feature for each video
    get_res_feature(config.frame_dir_tvsum, config.feature_dir_youtube)

if __name__ == '__main__':

    config = get_config(mode = 'train')
    
    #pre_process(config)

    feature_dir = config.feature_dir_youtube

    gt_dir = config.gt_dir_youtube
    
    train_loader = feature_loader(feature_dir, 'train')

    test_loader = feature_loader(feature_dir, 'test')

    gt_loader = gt_loader(gt_dir)

    lstmgan = LstmGan(config, train_loader, test_loader, gt_loader)

    lstmgan.build()

    #lstmgan.train()

    lstmgan.test()
