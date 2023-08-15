import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

class SizeInferenceModel(nn.Module):
    def __init__(self):
        super(SizeInferenceModel, self).__init__()

        self.state_dims = cfg.STATE_DIMS
        self.feat_dims = cfg.FEAT_DIMS
        self.size_dims = cfg.GRID_SHAPE[2] * cfg.GRID_SHAPE[3]

        ########################################################################
        # Define layers
        ########################################################################

        self.conv1 = nn.Conv2d(self.feat_dims[0], self.state_dims[2], kernel_size=3, dilation=2, padding=2)
        

        self.size_output = nn.Sequential(
            nn.Linear(self.state_dims[2], self.size_dims),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, input_feat, input_roi):
        
        rois = input_roi.view(self.state_dims[0] * self.state_dims[1])
        rois = rois.unsqueeze(1).repeat(1, 1, self.state_dims[2])
        #rois = rois.permute((2,1))
        rois = rois.view((self.state_dims[0], self.state_dims[1], self.state_dims[2]))

        size_hidden = F.relu(self.conv1(input_feat))

        #print(size_hidden.shape) #([1, 512, 15, 15])
        size_hidden = size_hidden.squeeze()
        #size_hidden = size_hidden.permute((2,1,0))
        #print(size_hidden.shape) #(torch.Size([15, 15, 512])

        #print(rois.shape) #([15, 15, 512])
        size_hidden = torch.mul(size_hidden, rois.permute((2,1,0)))
        size_hidden = F.adaptive_max_pool2d(size_hidden, output_size = 1)
        
        size_hidden = size_hidden.permute((2,1,0))
        size_output = self.size_output(size_hidden)
        return size_output
