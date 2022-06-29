# Modified from: https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py


from .NLayerDiscriminator import NLayerDiscriminator
import torch.nn as nn
import torch
from typing import Type, Optional


class MultiscaleDiscriminator(nn.Module):

    def __init__(self,
                 input_nc: int,
                 ndf: int = 64,
                 n_layer: int = 3,
                 use_sigmoid: bool = False,
                 num_D: int = 3,
                 norm_type="instance",
                 padding_type="zeros",
                 if_get_interm_feat=True
                 ):
        super(MultiscaleDiscriminator, self).__init__()
        self.__num_D = num_D
        self.__n_layers = n_layer

        for i in range(1, num_D + 1):
            netD = NLayerDiscriminator(input_nc=input_nc,
                                       ndf=ndf,
                                       n_layers=n_layer,
                                       use_sigmoid=use_sigmoid,
                                       norm_type=norm_type,
                                       padding_type=padding_type,
                                       if_get_interm_feat=if_get_interm_feat)
            setattr(self, "model{}".format(i), netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        result = []
        downsampled = x
        for i in range(1, self.__num_D):
            model = getattr(self, "model{}".format(i))
            model_out = model(downsampled)
            result.append(model_out)
            downsampled = self.downsample(downsampled)

        model = getattr(self, "model{}".format(self.__num_D))
        result.append(model(downsampled))
        return result

    @property
    def num_D(self) -> int:
        return self.__num_D

    @property
    def n_layer(self) -> int:
        return self.__n_layers