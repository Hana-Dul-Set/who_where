import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from conv_block import ConvBlock as conv_block

class CenterInferenceModel(nn.Module):
    def __init__(self):
        super(CenterInferenceModel, self).__init__()

        resolution = list(reversed(cfg.PREDICT_RESOLUTION))

        ########################################################################
        # Input
        ########################################################################
        self.conv1 = nn.Conv2d(in_channels=resolution[0] * 2, out_channels=64, kernel_size=7, stride=2)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_blocks = nn.Sequential(
            conv_block(64, [64, 64, 128], 3, stage=2, block='a'),
            conv_block(128, [64, 64, 128], 3, stage=3, block='a'),
            conv_block(128, [128, 128, 512], 3, stage=4, block='a')
        )

        self.cen_hidden = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, dilation=2, padding=2)
        )
        self.cen_reshape = nn.Flatten()

    def forward(self, input_img, input_lyo):
        ########################################################################
        # Forward pass
        ########################################################################
        inputs = torch.cat([input_img, input_lyo], dim=1)

        x = self.conv1(inputs)
        x = self.bn_conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv_blocks(x)

        feat = x

        cen_hidden = self.cen_hidden(feat)
        cen_hidden = self.cen_reshape(cen_hidden)

        cen_output = F.softmax(cen_hidden, dim=1)

        return feat, cen_output

# Instantiate the model
inference_model = CenterInferenceModel()

print(inference_model)