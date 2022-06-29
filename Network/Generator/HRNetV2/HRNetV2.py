from .HRNetV2Block import get_hrnet_backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.LayerHelper import LayerHelper
from .OCRBlock import SpatialGather_Module, SpatialOCR_Module
from collections.abc import Sequence
from .config import MODEL_CONFIGS
from typing import Union, Type, List


# from Network.utils.Layer import MCDropout2D, MCDropout


class HRNet(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, backbone: str, input_nc, output_nc, norm_type, padding_type, dropout_rate, dropout_layer):
        super(HRNet, self).__init__()
        self.backbone = get_hrnet_backbone(backbone=backbone,
                                           input_nc=input_nc,
                                           norm_type=norm_type,
                                           padding_type=padding_type,
                                           dropout_rate=dropout_rate,
                                           dropout_layer=dropout_layer)
        # extra added layers
        # in_channels = 720 # 48 + 96 + 192 + 384
        in_channels = sum(self.backbone.stage4_n_channels)
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_type),
            LayerHelper.get_norm_layer(num_features=in_channels, norm_type=norm_type),
            nn.ReLU(),
            # ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, output_nc, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)
        # out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return out


class HRNet_OCR(nn.Module):
    def __init__(self, backbone: str, input_nc, output_nc, norm_type, padding_type, dropout_rate, dropout_layer):
        super(HRNet_OCR, self).__init__()
        self.backbone = get_hrnet_backbone(backbone=backbone,
                                           input_nc=input_nc,
                                           norm_type=norm_type,
                                           padding_type=padding_type,
                                           dropout_rate=dropout_rate,
                                           dropout_layer=dropout_layer)
        in_channels = sum(self.backbone.stage4_n_channels)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1, padding_mode=padding_type),
            LayerHelper.get_norm_layer(num_features=512, norm_type=norm_type),
            nn.ReLU(),
        )
        self.ocr_gather_head = SpatialGather_Module()
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 norm_type=norm_type)
        self.cls_head = nn.Conv2d(512, output_nc, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_type),
            LayerHelper.get_norm_layer(num_features=in_channels, norm_type=norm_type),
            nn.ReLU(),
            nn.Conv2d(in_channels, output_nc, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        # out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        # out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return out, out_aux


class HRNetGenerator(nn.Module):
    def set_dropout_rate(self, p: float):
        assert p >= 0
        self.hrnet.backbone.drop1.p = p
        for drop_list in [self.hrnet.backbone.drop2,
                          self.hrnet.backbone.drop3,
                          self.hrnet.backbone.drop4]:
            for drop_layer in drop_list:
                drop_layer.p = p

    @staticmethod
    def config_names() -> list[str]:
        return list(MODEL_CONFIGS.keys())

    def __init__(self,
                 config: str,
                 input_nc,
                 output_nc,
                 img_dsize=None,
                 norm_type="group",
                 padding_type="reflect",
                 dropout_rate: float = 0.,
                 dropout_layer: Union[Type[nn.Dropout], Type[nn.Dropout2d]] = nn.Dropout2d,
                 ):
        super(HRNetGenerator, self).__init__()
        n_downsampling = 2
        ngf = 64
        self.hrnet = HRNet
        self.hrnet = self.hrnet(backbone=config,
                                input_nc=input_nc,
                                output_nc=ngf * (2 ** n_downsampling),
                                norm_type=norm_type,
                                padding_type=padding_type,
                                dropout_rate=dropout_rate,
                                dropout_layer=dropout_layer)

        self.upsampler = []
        self.upsampler_2 = None
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.upsampler += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1,
                                                  output_padding=1),
                               LayerHelper.get_norm_layer(num_features=int(ngf * mult / 2), norm_type=norm_type),
                               nn.ReLU(inplace=True)]
        self.upsampler += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3, padding_mode=padding_type),
                           nn.Tanh()]
        self.upsampler = nn.Sequential(*self.upsampler)

    def forward(self, x):
        x = self.hrnet(x)
        return self.upsampler(x)
