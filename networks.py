import torch
import math
from torch import nn
from collections import OrderedDict
class Extractor(nn.Module):
    def __init__(self, pretrained=False,*args, **kwargs) -> None:
        super(Extractor, self).__init__(*args, **kwargs)
        self.encoder1 = self.Unet_block(1, 32, "enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.Unet_block(32, 64, "enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.Unet_block(64, 128, "enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.Unet_block(128, 256, "enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.Unet_block(256, 512, "bottleneck")
        
        self.upconv4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.decoder4 = self.Unet_block(512, 256, "dec4")
        self.upconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.decoder3 = self.Unet_block(256, 128, "dec3")
        self.upconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.decoder2 = self.Unet_block(128, 64, "dec2")
        self.upconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.decoder1 = self.Unet_block(64, 32, "dec1")
        self.conv = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=1
        )
    def Unet_block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))
    


class Baseline_Unet(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    def weight_init_xavier_uniform(self, submodule):
        if isinstance(submodule, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(submodule.weight)
        elif isinstance(submodule, torch.nn.BatchNorm2d):
            submodule.weight.data.fill_(1.0)
            submodule.bias.data.zero_()
        elif isinstance(submodule, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(submodule.weight, a=math.sqrt(5))
    def __init__(self):
        super(Baseline_Unet, self).__init__()
        self.encoder = Extractor(pretrained=False)
        
        # for module in self.encoder.model_front.modules():
        #     self.weight_init_xavier_uniform(module)
        # self.color_linear.apply(self.weight_init_xavier_uniform)
    def forward(self, x):
        """ Forward propagation with input 'x'. """
        out = self.encoder(x)
        return out