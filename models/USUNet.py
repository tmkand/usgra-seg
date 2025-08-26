import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchinfo import summary
from utils.AbstractAndHelpers import MLNetwork
from utils.AbstractAndHelpers import DeviceSelection


# Implementation adapted from Aladdin Persson
# MIT License
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/model.py

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # do not need to learn bias since we are using batch normalization
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module, MLNetwork):
    """
    Takes normalized image inputs [0,1] torch.float32 in the shape (C, H, W) or (N, C, H, W)
    Masks with Data Type torch.long as Class Indices in the Shape (H, W) or (N, H, W)
    """
    def __init__(
            self, in_channels=1, out_channels=3, features=[64, 128, 256, 512],
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET = Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)      # features[-1] returns the last element of the list

        # Up part of UNET = Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Output Layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def summarize(self, batch_size=12, input_size=(1,512,512)):
        return str(summary(self, input_size=(batch_size,input_size[0],input_size[1],input_size[2]), depth=4, verbose=0))

if __name__ == "__main__":
    device = DeviceSelection().device
    model = UNET(in_channels=1, out_channels=3, features=[64, 128, 256, 512]).to(device)
    print(model.summarize(batch_size=12, input_size=(1,512,512)))