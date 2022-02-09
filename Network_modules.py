import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RN18_Encoder(nn.Module):
    def __init__(self):
        super(RN18_Encoder, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)

        # to get feature map -- model without 'avgpool' and 'fc'
        self.features = nn.Sequential()
        for name, module in self.backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, pool=True):
        x = self.features(x)
        if pool:
            x = self.gap(x)
            x = torch.flatten(x, 1)
        return x

class UNet_Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(UNet_Decoder, self).__init__()
        self.deconv_1 = Unet_UpBlock(512, 512)
        self.deconv_2 = Unet_UpBlock(512, 512)
        self.deconv_3 = Unet_UpBlock(512, 512)
        self.deconv_4 = Unet_UpBlock(512, 256)
        self.deconv_5 = Unet_UpBlock(256, 128)
        self.deconv_6 = Unet_UpBlock(128, 64)
        self.deconv_7 = Unet_UpBlock(64, 32)
        self.final_image = nn.Sequential(*[nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)])

    def forward(self, x):
        x = x.view(-1, 512, 1, 1)
        x = self.deconv_1(x)  # 2
        x = self.deconv_2(x)  # 4
        x = self.deconv_3(x)  # 8
        x = self.deconv_4(x)  # 16
        x = self.deconv_5(x)  # 32
        x = self.deconv_6(x)  # 64
        x = self.deconv_7(x)  # 128
        x = self.final_image(x)  # 256
        return x


class Unet_UpBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc):
        super(Unet_UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(outer_nc),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    tns = torch.randn((4,3,256,256))
    model_enc = RN18_Encoder()
    fmap = model_enc(tns, pool=False)
    print(f"fmap shape: {fmap.shape}")
    model_dec = UNet_Decoder()
    recons = model_dec(fmap)
    print(f"recons shape: {recons.shape}")
