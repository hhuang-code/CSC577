from pathlib import Path
from PIL import Image
from torchvision.datasets.folder import default_loader

import os
import h5py
import numpy as np
import subprocess

from networks.ResNet import ResNet

import pdb

"""
extrace frames from each video
"""
def get_frames(video_dir, frame_dir):
    print(video_dir)
    pdb.set_trace()
    for video in sorted(video_dir.glob('*.*')):
        tokens = str(video).split('/')
        subfolder = (tokens[-1].split('.'))[0]
        cur_frame_dir = frame_dir.joinpath(subfolder)
        if not os.path.exists(str(cur_frame_dir)):
            os.makedirs(str(cur_frame_dir))
        # call ffmpeg
        subprocess.call(['ffmpeg', '-i', str(video), '-vf', 'fps = 2', str(cur_frame_dir) + '/' + 'thumb%05d.jpg', '-hide_banner'])
    
"""
extract ResNet feature, one video per feature file
"""
def get_res_feature(frame_dir, feature_dir):

    # build resnet class
    resnet = ResNet(224)

    for sub_frame_dir in sorted(frame_dir.glob('*/')):
        video_feature = np.array([])
        for frame_filename in sorted(sub_frame_dir.glob('*.jpg')):   # for each frame image in dir_path
            frame = default_loader(str(frame_filename))
            print(frame_filename)
            # extract ResNet feature
            res_conv5, res_pool5 = resnet(frame)
            # gpu variable -> cpu variable -> tensor -> numpy array -> 1D array
            frame_feature = res_pool5.cpu().data.numpy().flatten()
            if video_feature.size == 0:
                video_feature = np.hstack((video_feature, frame_feature))
            else:
                video_feature = np.vstack((video_feature, frame_feature))
            print(video_feature.shape)
        
        if not os.path.exists(str(feature_dir)):
            os.makedirs(str(feature_dir))
        h5_filename = str(feature_dir) + '/' + (str(sub_frame_dir).split('/'))[-1] + '.h5'
        h5file = h5py.File(h5_filename, 'w')
        h5file.create_dataset('pool5', data = video_feature)
        h5file.close()
