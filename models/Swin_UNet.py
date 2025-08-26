import torch
import torch.nn as nn
import timm


class SwinUNet(nn.Module):
    def __init__(self, in_chans=1, num_classes=3, backbone='swin_base_patch4_window7_224'):
        super().__init__()

        # Pretrained Swin Transformer encoder (expects 3-channel input)
        self.encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            img_size=512
        )
        encoder_channels = self.encoder.feature_info.channels()  # e.g. [128, 256, 512, 1024]

        # Optional 1→3 channel conversion for grayscale input
        if in_chans == 1:
            self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)
        else:
            self.input_adapter = nn.Identity()

        # Decoder: progressively upsample and fuse with skip connections
        self.up3 = self._upsample_block(encoder_channels[3], encoder_channels[2])
        self.up2 = self._upsample_block(encoder_channels[2], encoder_channels[1])
        self.up1 = self._upsample_block(encoder_channels[1], encoder_channels[0])
        self.final_up = nn.ConvTranspose2d(encoder_channels[0], 64, kernel_size=2, stride=2)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.out_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.input_adapter(x)  # [B, 1, H, W] → [B, 3, H, W]
        feats = self.encoder(x)  # list of feature maps: [x1, x2, x3, x4]

        # Ensure features are [B, C, H, W]
        feats = [f.permute(0, 3, 1, 2) if f.shape[1] != self.encoder.feature_info.channels()[i] else f
                 for i, f in enumerate(feats)]

        x4 = feats[-1]  # 1/32
        x3 = self.up3(x4) + feats[-2]  # 1/16
        x2 = self.up2(x3) + feats[-3]  # 1/8
        x1 = self.up1(x2) + feats[-4]  # 1/4

        out = self.final_up(x1)  # back to 256x256
        out = self.out_up(out) # 512x512
        out = self.out_conv(out)  # [B, num_classes, H, W]
        return out