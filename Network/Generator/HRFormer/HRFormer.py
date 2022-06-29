import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.LayerHelper import LayerHelper
from .HRFormerBlock import get_hrt_backbone
from .config import MODEL_CONFIGS
from typing import Type, Union


class HRT_V3(nn.Module):
    def __init__(self, configer: str, input_nc, output_nc, norm_type, padding_type, dropout_rate, dropout_layer):
        super(HRT_V3, self).__init__()
        self.configer = configer
        # self.num_classes = self.configer.get("data", "num_classes")
        self.backbone = get_hrt_backbone(configer,
                                         input_nc=input_nc,
                                         norm_type=norm_type,
                                         padidng_type=padding_type,
                                         dropout_rate=dropout_rate,
                                         dropout_layer=dropout_layer)

        in_channels = 1170 if configer.startswith("hrt_base") else 480
        hidden_dim = 512
        group_channel = math.gcd(in_channels, hidden_dim)
        # self.conv3x3 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels,
        #         hidden_dim,
        #         kernel_size=7,
        #         stride=1,
        #         padding=3,
        #         groups=group_channel,
        #     ),
        #     ModuleHelper.BNReLU(
        #         hidden_dim, bn_type=self.configer.get("network", "bn_type")
        #     ),
        # )
        # self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        # self.ocr_distri_head = SpatialOCR_Module(
        #     in_channels=hidden_dim,
        #     key_channels=hidden_dim // 2,
        #     out_channels=hidden_dim,
        #     scale=1,
        #     dropout=0.05,
        #     bn_type=self.configer.get("network", "bn_type"),
        # )
        # self.cls_head = nn.Conv2d(
        #     hidden_dim, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        # )
        self.aux_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=group_channel,
                padding_mode=padding_type,
            ),
            LayerHelper.get_norm_layer(num_features=hidden_dim, norm_type=norm_type),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                output_nc,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
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

        # feats = self.conv3x3(feats)
        #
        # context = self.ocr_gather_head(feats, out_aux)
        # feats = self.ocr_distri_head(feats, context)
        #
        # out = self.cls_head(feats)

        # out_aux = F.interpolate(
        #     out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        # )
        # out = F.interpolate(
        #     out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        # )
        # return out_aux, out
        return out_aux


class HRTGenerator(nn.Module):

    def set_dropout_rate(self, p: float):
        assert p >= 0
        self.backbone.backbone.drop1.p = p
        for drop_list in [self.backbone.backbone.drop2,
                          self.backbone.backbone.drop3,
                          self.backbone.backbone.drop4]:
            for drop_layer in drop_list:
                drop_layer.p = p

    @staticmethod
    def config_names() -> list[str]:
        return list(MODEL_CONFIGS.keys())

    def __init__(self,
                 backbone: str,
                 input_nc,
                 output_nc,
                 img_dsize=None,
                 norm_type="group",
                 padding_type="reflect",
                 dropout_rate: float = 0.,
                 dropout_layer: Union[Type[nn.Dropout], Type[nn.Dropout2d]] = nn.Dropout2d,
                 hr_decoder=False,
                 ):
        super(HRTGenerator, self).__init__()
        self.hr_decoder = hr_decoder
        n_downsampling = 2
        ngf = 64
        # self.hrnet = HRNet_OCR if self.ocr else HRNet

        self.backbone = HRT_V3(configer=backbone,
                               input_nc=input_nc,
                               output_nc=ngf * (2 ** n_downsampling),
                               norm_type=norm_type,
                               padding_type=padding_type,
                               dropout_rate=dropout_rate,
                               dropout_layer=dropout_layer
                               )

        self.upsampler = []
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



    def forward(self, x, cond_tensor=None):
        y = self.backbone(x)
        if self.regression:
            y = self.pool(y).view(y.shape[0], -1)
            if cond_tensor is not None:
                y = torch.concat([y, cond_tensor], dim=-1)
            y = self.fc(y)
        else:
            y = self.upsampler(y)
        return y
        # if self.ocr:
        #     out, out_aux = self.hrnet(x)
        #     out = self.upsampler(out)
        #     out_aux = self.upsampler_2(out_aux)
        #     return out, out_aux
        # else: