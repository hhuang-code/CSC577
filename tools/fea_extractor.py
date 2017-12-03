from pathlib import Path
from PIL import Image
from torchvision.datasets.folder import default_loader

import h5py
import numpy as np

from networks.ResNet import ResNet

import pdb

"""
extract ResNet feature, and one video per feature file
"""
def get_res_feature():
    
    dir_path = Path('/home/aaron/Documents/Courses/577/dataset/frame/Youtube/v71')

    resnet = ResNet(224)

    video_feature = np.array([])
    #pdb.set_trace()
    for frame_path in sorted(dir_path.glob('*.jpg')):   # for each frame image in dir_path
        frame = default_loader(str(frame_path))
        print(frame_path)
        # extract ResNet feature
        res_conv5, res_pool5 = resnet(frame)
        # gpu variable -> cpu variable -> tensor -> numpy array -> 1D array
        frame_feature = res_pool5.cpu().data.numpy().flatten()
        if video_feature.size == 0:
            video_feature = np.hstack((video_feature, frame_feature))
        else:
            video_feature = np.vstack((video_feature, frame_feature))
        print(video_feature.shape)

    h5file = h5py.File('/home/aaron/Documents/Courses/577/dataset/feature/Youtube/v71.h5', 'w')
    h5file.create_dataset('pool5', data = video_feature)
    h5file.close()
