from pathlib import Path
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable

import h5py
import numpy as np
from tqdm import tqdm

from tools.fea_extractor import get_res_feature
from tools.data_loader import feature_loader
from networks.Summarizer import Summarizer

import pdb

if __name__ == '__main__':

    # generate ResNet feature
    get_res_feature()

    """
    #pdb.set_trace()
    fea_dir = Path('/home/aaron/Documents/Courses/577/dataset/feature/Youtube')
    
    train_loader = feature_loader(fea_dir, 'train')

    for batch_idx, feature in enumerate(tqdm(train_loader, desc = 'Batch', leave = False)):
        if batch_idx == 1:
            summarizer = Summarizer(2048, 2048).cuda()
            # feature: (1, seq_len, input_size) -> (seq_len, 1, input_size)
            feature = Variable(feature.view(feature.shape[1], -1, feature.shape[2]))
            weighted_feature = summarizer(feature.cuda())
            print(weighted_feature)
    """
