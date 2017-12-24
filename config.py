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
    parser.add_argument('--hidden_size', type = int, default = 1024)
    parser.add_argument('--num_layers', type = int, default = 2)
    parser.add_argument('--summary_rate', type = int, default = 0.2)
    
    # train
    parser.add_argument('--n_epochs', type = int, default = 50)
    parser.add_argument('--sum_learning_rate', type = float, default = 1e-4)
    parser.add_argument('--dis_learning_rate', type = float, default = 1e-5)
    parser.add_argument('--dis_start_batch', type = int, default = 15)
    
    # log
    parser.add_argument('--log_dir', type = str, default = Path('/localdisk/log/'))
    parser.add_argument('--detail_flag', type = bool, default = True)

    # dataset path
    parser.add_argument('--video_dir_youtube', type = str, default = Path('/localdisk/577/dataset/video/Youtube'))
    parser.add_argument('--frame_dir_youtube', type = str, default = Path('/localdisk/577/dataset/frame/Youtube'))
    parser.add_argument('--feature_dir_youtube', type = str, default = Path('/localdisk/577/dataset/feature/Youtube'))
    parser.add_argument('--gt_dir_youtube', type = str, default = Path('/localdisk/577/dataset/gt/Youtube'))
  
    parser.add_argument('--video_dir_tvsum', type = str, default = Path('/localdisk/577/dataset/video/TVSum'))
    parser.add_argument('--frame_dir_tvsum', type = str, default = Path('/localdisk/577/dataset/frame/TVSum'))
    parser.add_argument('--feature_dir_tvsum', type = str, default = Path('/localdisk/577/dataset/feature/TVSum'))
    parser.add_argument('--gt_dir_tvsum', type = str, default = Path('/localdisk/577/dataset/gt/TVSum'))
   
    parser.add_argument('--video_dir_summe', type = str, default = Path('/localdisk/577/dataset/video/SumMe'))
    parser.add_argument('--frame_dir_summe', type = str, default = Path('/localdisk/577/dataset/frame/SumMe'))
    parser.add_argument('--feature_dir_summe', type = str, default = Path('/localdisk/577/dataset/feature/SumMe'))
    parser.add_argument('--gt_dir_summe', type = str, default = Path('/localdisk/577/dataset/gt/SumMe'))
   
    parser.add_argument('--video_dir_openvideo', type = str, default = Path('/localdisk/577/dataset/video/OpenVideo'))
    parser.add_argument('--frame_dir_openvideo', type = str, default = Path('/localdisk/577/dataset/frame/OpenVideo'))
    parser.add_argument('--feature_dir_openvideo', type = str, default = Path('/localdisk/577/dataset/feature/OpenVideo'))
    parser.add_argument('--gt_dir_openvideo', type = str, default = Path('/localdisk/577/dataset/gt/OpenVideo'))
    
    # mode save path
    parser.add_argument('--model_save_dir', type = str, default = Path('/localdisk/577/model/'))
    
    args = parser.parse_args()

    # namespace -> dictionary
    args = vars(args)
    args.update(kwargs)

    return Config(**args)
