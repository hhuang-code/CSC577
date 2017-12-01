import torch
import torch.nn as nn
import numpy as np

import pdb

"""
calculate loss
Args:
    dpp_weight: the weight of dpp loss over total loss
"""
class LOSS(nn.Module):
    def __init__(self, dpp_weight):
        super(LOSS, self).__init__()
        self.dpp_weight = dpp_weight

    """
    Args:
        pred: 0-1 prediction by the model
        bin_label: binary ground truth, 0 - non keyframe, 1 - keyframe
        idx_label: keyframe index
        L: dpp kernel matrix
    """
    def forward(self, pred, bin_label, idx_label, L):
        pdb.set_trace()
        # calculate dpp loss
        L = L.data.numpy()
        Lz = L[idx_label, :][:, idx_label]
        identity_mat = torch.eye(pred.data.shape[0])
        dpp_loss = -np.log(np.linalg.det(Lz)) - np.log(np.linalg.det(L + identity_mat.numpy()))
        
        # calculate prediction loss
        pred_loss = torch.mean((pred.view(pred.size(0)).data - torch.Tensor(bin_label)) ** 2)

        if np.isnan(dpp_loss):
            loss = pred_loss + self.dpp_weight * np.linalg.det(Lz + np.eye(Lz.shape[0]))
        else:
            loss = pred_loss + self.dpp_weight * dpp_loss
    
        return loss
