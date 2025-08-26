import torch
import torch.nn as nn
from torchinfo import summary
from utils.AbstractAndHelpers import MLNetwork
from utils.AbstractAndHelpers import DeviceSelection

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


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.zeros(1, height * width, channels))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, x):
        return x + self.pos_encoding


class UNET_ViTBottleneck(nn.Module, MLNetwork):
    def __init__(self, in_channels=1, out_channels=3, base_dim=64, img_size=512):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_dim)
        self.enc2 = DoubleConv(base_dim, base_dim*2)
        self.enc3 = DoubleConv(base_dim*2, base_dim*4)
        self.enc4 = DoubleConv(base_dim*4, base_dim*8)

        self.pool = nn.MaxPool2d(2)

        #Bottleneck
        self.pos_encoding = PositionalEncoding2D(channels=base_dim * 16, height=img_size // 16, width=img_size // 16)
        self.bottleneck_conv = DoubleConv(base_dim*8, base_dim*16)
        self.transformer = TransformerBlock(dim=base_dim*16)
        self.img_size = img_size // 16

        #Decoder
        self.up1 = nn.ConvTranspose2d(base_dim*16, base_dim*8, 2, 2)
        self.dec1 = DoubleConv(base_dim*16, base_dim*8)
        self.up2 = nn.ConvTranspose2d(base_dim*8, base_dim*4, 2, 2)
        self.dec2 = DoubleConv(base_dim*8, base_dim*4)
        self.up3 = nn.ConvTranspose2d(base_dim*4, base_dim*2, 2, 2)
        self.dec3 = DoubleConv(base_dim*4, base_dim*2)
        self.up4 = nn.ConvTranspose2d(base_dim*2, base_dim, 2, 2)
        self.dec4 = DoubleConv(base_dim*2, base_dim)

        self.final_conv = nn.Conv2d(base_dim, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Bottleneck
        x_b = self.bottleneck_conv(self.pool(x4))
        B, C, H, W = x_b.shape
        x_flat = x_b.flatten(2).permute(0, 2, 1) # (B, N=H*W, C)
        x_flat = self.pos_encoding(x_flat)
        x_flat = self.transformer(x_flat)
        x_b = x_flat.permute(0, 2, 1).reshape(B, C, H, W)

        # Decoder
        x = self.up1(x_b)
        x = self.dec1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.dec4(torch.cat([x, x1], dim=1))

        return self.final_conv(x)

    def summarize(self, batch_size=12, input_size=(1, 512, 512)):
        return str(
            summary(self, input_size=(batch_size, input_size[0], input_size[1], input_size[2]), depth=4, verbose=0))

if __name__ == "__main__":
    device = DeviceSelection().device
    model = UNET_ViTBottleneck(in_channels=1, out_channels=3).to(device)
    print(model.summarize(batch_size=12, input_size=(1, 512, 512)))
