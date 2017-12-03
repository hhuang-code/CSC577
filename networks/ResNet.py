from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable

import pdb

"""
Rescale a image so that the smaller edge is a given size.
For ResNet, the image should be at leaset 224 * 224.
"""
class Rescale(object):
    """
    Args:
        scale_size: the smaller edge of the image will be rescaled to scale_size
    """
    def __init__(self, scale_size):
        self.scale_size = scale_size

    """
    Args:
        image: PIL.image, image to rescale
    """
    def __call__(self, image):
        cur_width, cur_height = image.size
        min_edge = min(cur_width, cur_height)
        
        # rescale ratio
        ratio = self.scale_size / min_edge

        # float to int
        if cur_width * ratio > self.scale_size:
            new_width = int(cur_width * ratio)
        else:
            new_width = self.scale_size

        if cur_height * ratio >= self.scale_size:
            new_height = int(cur_height * ratio)
        else:
            new_height = self.scale_size

        # rescale
        new_image = image.resize((new_width, new_height), resample = Image.BILINEAR)

        return new_image

class ResNet(nn.Module):
    """
    Args:
        fea_type: string, resnet101 or resnet 152
        input_size: the smaller edge of input image should be input_size
    """
    def __init__(self, input_size, fea_type = 'resnet101'):
        super(ResNet, self).__init__()
        self.fea_type = fea_type
        # rescale and normalize transformation
        self.transform = transforms.Compose([
                Rescale(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        if fea_type == 'resnet101':
            resnet = models.resnet101(pretrained = True)    # dim of pool5 is 2048
        elif fea_type == 'resnet152':
            resnet = models.resnet152(pretrained = True)
        else:
            raise "No such ResNet!"

        resnet.float()
        resnet.cuda()
        resnet.eval()

        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[: -2])
        self.pool5 = module_list[-2]

    # rescale and normalize image, then pass it through ResNet
    def forward(self, x):
        x = self.transform(x)
        x = x.unsqueeze(0)  # reshape the single image s.t. it has a batch dim
        #pdb.set_trace()
        x = Variable(x).cuda()
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)
        
        return res_conv5, res_pool5 
        
