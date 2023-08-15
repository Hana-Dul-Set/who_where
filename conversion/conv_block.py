import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stage, block, strides=(2, 2)):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = out_channels

        self.conv_name_base = f'res{stage}{block}_branch'
        self.bn_name_base = f'bn{stage}{block}_branch'

        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=(1, 1), stride=strides, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(filters1)
        
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(filters2)
        
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(filters3)
        
        self.shortcut_conv = nn.Conv2d(in_channels, filters3, kernel_size=(1, 1), stride=strides, padding=0, bias=True)
        self.shortcut_bn = nn.BatchNorm2d(filters3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        residual = self.shortcut_conv(residual)
        residual = self.shortcut_bn(residual)

        x += residual
        x = self.relu(x)

        return x

# Example usage
input_tensor = torch.randn(64, 64, 56, 56)  # Example input tensor shape (batch_size, channels, height, width)
conv_block = ConvBlock(in_channels=64, out_channels=[64, 64, 256], kernel_size=3, stage=2, block='a')
output_tensor = conv_block(input_tensor)