#  Copyright (c) 2021 by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from .DAFormerDecoder import DAFormerHead
from .MixVisionTransformer import MixVisionTransformer
import torch.nn as nn
from ...utils.LayerHelper import LayerHelper


class DAFormerGenerator(nn.Module):

    @staticmethod
    def config_names() -> list[str]:
        return list(MixVisionTransformer.configs.keys())

    def __init__(self,
                 config: str,
                 img_dsize: tuple[int, int],
                 input_nc: int,
                 output_nc: int,
                 dropout_rate: float = 0.,
                 norm_type="group",
                 padding_type="reflect"):
        super(DAFormerGenerator, self).__init__()
        n_downsampl = 2
        ngf = 64

        encoder = MixVisionTransformer.get_model(config=config,
                                                 img_dsize=img_dsize,
                                                 input_nc=input_nc,
                                                 dropout_rate=dropout_rate,
                                                 # output_nc=64 * (2 ** n_downsampl),
                                                 padding_type=padding_type)
        decoder = DAFormerHead.get_da_former(num_classes=64 * (2 ** n_downsampl),
                                             in_channels=encoder.embed_dims,
                                             padding_type=padding_type,
                                             dropout_rate=dropout_rate,
                                             norm_type=norm_type)

        final_layers = []
        ### upsample
        for i in range(n_downsampl):
            mult = 2 ** (n_downsampl - i)
            final_layers += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                                output_padding=1, bias=True),
                             LayerHelper.get_norm_layer(num_features=int(ngf * mult / 2), norm_type=norm_type),
                             # norm_layer(int(ngf * mult / 2)),
                             nn.ReLU(inplace=True)]
        final_layers += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3, padding_mode=padding_type),
                         nn.Tanh()]
        self.model = nn.Sequential(encoder, decoder, *final_layers)

    def forward(self, x):
        # y = self.encoder(x)
        # return self.decoder(y)
        return self.model(x)
