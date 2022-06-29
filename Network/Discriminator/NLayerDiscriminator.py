# Modified from: https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from ..utils.LayerHelper import LayerHelper


class NLayerDiscriminator(nn.Module):
    @staticmethod
    def names() -> [str, ...]:
        return [NLayerDiscriminator.__name__]

    def __init__(self,
                 input_nc: int,
                 ndf: int = 64,
                 n_layers: int = 3,
                 norm_type: Optional[str] = None,
                 use_sigmoid: bool = False,
                 padding_type: str = "zeros",
                 if_get_interm_feat = True):
        super(NLayerDiscriminator, self).__init__()
        self.__n_layers = n_layers
        self.__if_get_interm_feat = if_get_interm_feat

        kw = 4
        # padw = int(np.ceil((kw - 1.0) / 2))
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, padding_mode=padding_type),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, padding_mode=padding_type),
                LayerHelper.get_norm_layer(num_features=nf, norm_type=norm_type),
                # norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, padding_mode=padding_type),
            LayerHelper.get_norm_layer(num_features=nf, norm_type=norm_type),
            # norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw, padding_mode=padding_type)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if self.__if_get_interm_feat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.__if_get_interm_feat:
            res = [input]
            for n in range(self.__n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return [self.model(input)]
