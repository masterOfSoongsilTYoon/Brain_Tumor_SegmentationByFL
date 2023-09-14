from typing import Dict, Optional
import torch
import math
from torch import Tensor, nn
from collections import OrderedDict
import torchvision
from typing import Dict
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, pretrained=False,*args, **kwargs) -> None:
        super(Extractor, self).__init__(*args, **kwargs)
        self.encoder1 = self.Unet_block(3, 32, "enc1")
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
    
class Classfier(nn.Module):
    def __init__(self, in_channel, out_channel,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channel =in_channel
        self.out_channel = out_channel
        self.module = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, 0, 1, bias=False),
                                    nn.Sigmoid()
                                )
    def forward(self, x):
        out=self.module(x)
        return out
class DeeplabRes(nn.Module):
    def __init__(self, module,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = module
        
    def forward(self, x):
        x= self.module(x)
        return {"out": x}
class OpenExtractor(nn.Module):
    def __init__(self, in_channel=1,backbone="101"):
        super(OpenExtractor, self).__init__()
        if backbone =="101":
            self.module = list(torchvision.models.resnet101().children())[:-2]
            if in_channel ==1:
                self.seq = nn.Sequential(nn.Conv2d(1,3,1,1,0,1,bias=False),
                                          *self.module)
            else:
                self.seq = nn.Sequential(*self.module)
            self.backbone = DeeplabRes(self.seq)
            self.classfier = Classfier(2048, 1)
        elif backbone =="swin":
            self.module = list(torchvision.models.swin_v2_t().children())[:-2]
            if in_channel ==1:
                self.seq = nn.Sequential(nn.Conv2d(1,3,1,1,0,1,bias=False),
                                          *self.module)
            else:
                self.seq = nn.Sequential(*self.module)
            self.backbone = DeeplabRes(self.seq)
            self.classfier = Classfier(768, 1)
        elif backbone =="efficient":
            self.module = list(torchvision.models.efficientnet_v2_s().children())[:-2]
            if in_channel ==1:
                self.seq = nn.Sequential(nn.Conv2d(1,3,1,1,0,1,bias=False),
                                          *self.module)
            else:
                self.seq = nn.Sequential(*self.module)
            self.backbone = DeeplabRes(self.seq)
            self.classfier = Classfier(1280, 1)
        self.deeplabV3 = torchvision.models.segmentation.DeepLabV3(self.backbone, self.classfier, None)
        
    def forward(self, x):
        out=self.deeplabV3(x)
        return out


class BrainClassifier(nn.Module):
    def weight_init_xavier_uniform(self, submodule):
        if isinstance(submodule, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(submodule.weight)
        elif isinstance(submodule, torch.nn.BatchNorm2d):
            submodule.weight.data.fill_(1.0)
            submodule.bias.data.zero_()
        elif isinstance(submodule, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(submodule.weight, a=math.sqrt(5))
    
    def __init__(self, in_channel=1,backbone="101"):
        super(BrainClassifier, self).__init__()
        if backbone =="101":
            self.module = list(torchvision.models.resnet101().children())[:-2]
            if in_channel ==1:
                self.seq = nn.Sequential(nn.Conv2d(1,3,1,1,0,1,bias=False),
                                          *self.module,
                                          nn.AdaptiveAvgPool2d((1,1))
                                          )
            else:
                self.seq = nn.Sequential(*self.module,
                                          nn.AdaptiveAvgPool2d((1,1))
                )
                                          
            self.backbone = self.seq
            self.fcl = nn.Sequential(nn.Linear(2048,5),
                                     nn.Softmax(dim=1)
                                     )
        elif backbone =="swin":
            self.module = list(torchvision.models.swin_v2_t().children())[:-2]
            if in_channel ==1:
                self.seq = nn.Sequential(nn.Conv2d(1,3,1,1,0,1,bias=False),
                                          *self.module,
                                          nn.AdaptiveAvgPool2d((1,1))
                                          )
            else:
                self.seq = nn.Sequential(*self.module,
                                          nn.AdaptiveAvgPool2d((1,1))
                )
            self.backbone = self.seq
            self.fcl = nn.Sequential(nn.Linear(768,5),
                                     nn.Softmax(dim=1)
                                     )
        elif backbone =="efficient":
            self.module = list(torchvision.models.efficientnet_v2_s().children())[:-2]
            if in_channel ==1:
                self.seq = nn.Sequential(nn.Conv2d(1,3,1,1,0,1,bias=False),
                                          *self.module,
                                          nn.AdaptiveAvgPool2d((1,1))
                                          )
            else:
                self.seq = nn.Sequential(*self.module,
                                          nn.AdaptiveAvgPool2d((1,1))
                )
                                          
            self.backbone = self.seq
            self.fcl = nn.Sequential(nn.Linear(1280,5),
                                     nn.Softmax(dim=1)
                                     )
        elif backbone =="vgg":
            self.module = list(torchvision.models.vgg16_bn().children())[:-2]
            if in_channel ==1:
                self.seq = nn.Sequential(nn.Conv2d(1,3,1,1,0,1,bias=False),
                                          *self.module,
                                          nn.AdaptiveAvgPool2d((1,1))
                                          )
            else:
                self.seq = nn.Sequential(*self.module,
                                          nn.AdaptiveAvgPool2d((1,1))
                )
                                          
            self.backbone = self.seq
            self.fcl = nn.Sequential(nn.Linear(512,5),
                                     nn.Softmax(dim=1)
                                     )
        
        
        self.backbone.apply(self.weight_init_xavier_uniform)
        self.fcl.apply(self.weight_init_xavier_uniform)
    def forward(self, x):
        out = self.backbone(x)
        out = out.squeeze()
        out = self.fcl(out)
        # out = torch.argmax(out, dim=1)
        return out


class Basconv(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, is_batchnorm = False, kernel_size = 3, stride = 1, padding=1):
        super(Basconv, self).__init__()
        
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.ReLU(inplace=True))

        # initialise the blocks
        # for m in self.children():
        #     weight_init_xavier_uniform(m)
    
    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        return x

class UnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm, n=2, kernel_size = 3, stride=1, padding=1):
        super(UnetConv, self).__init__()
        self.n = n    

        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_channels = out_channels

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_channels = out_channels

        # initialise the blocks
        # for m in self.children():
        #     weight_init_xavier_uniform(m)

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)
        return x

class UnetUp(nn.Module):
    def __init__(self,in_channels, out_channels, is_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv(in_channels+(n_concat-2)* out_channels, out_channels, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('UnetConv') != -1: continue
        #     weight_init_xavier_uniform(m)

    def forward(self, inputs0,*input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0,input[i]], 1)
        return self.conv(outputs0)

class UnetUp4(nn.Module):
    def __init__(self,in_channels, out_channels, is_deconv, n_concat=2):
        super(UnetUp4, self).__init__()
        self.conv = UnetConv(in_channels+(n_concat-2)* out_channels, out_channels, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=6, stride=4, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('UnetConv') != -1: continue
        #     weight_init_xavier_uniform(m)

    def forward(self, inputs0,*input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0,input[i]], 1)
        return self.conv(outputs0)
    
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h
    
class GloRe_Unit(nn.Module):

    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = Basconv(num_in, self.num_s, is_batchnorm = True, kernel_size=kernel_size, padding=padding)  
        # generate projection and inverse projection functions
        self.conv_proj = Basconv(num_in, self.num_n, is_batchnorm = True,kernel_size=kernel_size, padding=padding)   
        self.conv_reproj = Basconv(num_in, self.num_n, is_batchnorm = True,kernel_size=kernel_size, padding=padding)  
        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)   
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)  
        # fusion
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1), 
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in) 

    def forward(self, x):
        batch_size = x.size(0)
        # generate projection and inverse projection matrices
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1) 
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
        x_rproj_reshaped = self.conv_reproj(x).view(batch_size, self.num_n, -1)
        # project to node space
        x_n_state1 = torch.bmm(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1)) 
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        # graph convolution
        x_n_rel1 = self.gcn1(x_n_state2)  
        x_n_rel2 = self.gcn2(x_n_rel1)
        # inverse project to original space
        x_state_reshaped = torch.bmm(x_n_rel2, x_rproj_reshaped)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])
        # fusion
        out = x + self.blocker(self.fc_2(x_state))

        return out
    
class  MGR_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Module, self).__init__()

        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

        self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
        self.f1 = Basconv(in_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)

        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        self.g2 = self.glou2(self.x2)
        self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        self.g3= self.glou3(self.x3)
        self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)

        return self.f1(out)

class MGUNet_2(nn.Module):
    def __init__(self, in_channels=1, n_classes=11, feature_scale=4, is_deconv=True, is_batchnorm=True):  ##########
        super(MGUNet_2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # encoder
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.mgb =  MGR_Module(filters[2], filters[3])

        self.center = UnetConv(filters[2], filters[3], self.is_batchnorm)

        # decoder
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        if n_classes ==1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax2d()
        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  
        maxpool1 = self.maxpool1(conv1) 
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)  
        conv3 = self.conv3(maxpool2)  
        maxpool3 = self.maxpool3(conv3)  
        feat_sum = self.mgb(maxpool3) 
        center = self.center(feat_sum)  
        up3 = self.up_concat3(center, conv3) 
        up2 = self.up_concat2(up3, conv2) 
        up1 = self.up_concat1(up2, conv1)
        final_1 = self.final_1(up1)
        final_1 = self.sigmoid(final_1)
        return final_1

class Baseline_net(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    def weight_init_xavier_uniform(self, submodule):
        if isinstance(submodule, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(submodule.weight)
        elif isinstance(submodule, torch.nn.BatchNorm2d):
            submodule.weight.data.fill_(1.0)
            submodule.bias.data.zero_()
        elif isinstance(submodule, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(submodule.weight, a=math.sqrt(5))
    def __init__(self,in_channel= 1,mode="unet", data="OCT", backbone="101"):
        super(Baseline_net, self).__init__()
        if mode == "unet":
            if in_channel==1:
                self.encoder = nn.Sequential(nn.Conv2d(1,3,1,1,0,1,bias=False),
                                             Extractor(pretrained=False))
            else:
                self.encoder =  Extractor(pretrained=False)
        elif mode =="deeplab":
            self.encoder = OpenExtractor(in_channel=in_channel, backbone=backbone)
        elif mode =="mgunet":
            self.encoder = MGUNet_2(in_channel, 1, 2)
        self.mode = mode
        self.data = data
        for module in self.encoder.modules():
            self.weight_init_xavier_uniform(module)
        
    def forward(self, x):
        """ Forward propagation with input 'x'. """
        out = self.encoder(x)
        if self.mode=="unet":
            return out
        elif self.mode =="deeplab":
            return out["out"]
        elif self.mode =="mgunet":
            return out
        