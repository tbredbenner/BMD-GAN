# Modified from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

import torch.nn as nn
from ..utils.LayerHelper import LayerHelper


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    @staticmethod
    def config_names() -> list[str]:
        return ["resnet_d2", "resnet_d3", "resnet_d4"]

    def __init__(self,
                 config,
                 input_nc,
                 output_nc,
                 img_dsize=None,
                 norm_type="instance",
                 dropout=0.,
                 dropout_block_list=None,
                 padding_type='reflect'):
        if config == "resnet_d2":
            n_downsampling = 2
        elif config == "resnet_d3":
            n_downsampling = 3
        elif config == "resnet_d4":
            n_downsampling = 4
        else:
            raise NotImplementedError(f"Unknown config {config}")
        ngf = 64
        n_blocks = 9



        if dropout_block_list is None:
            dropout_block_list = list(range(n_blocks))
        super(ResnetGenerator, self).__init__()
        activation = nn.ReLU(True)
        use_bias = True

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 LayerHelper.get_norm_layer(num_features=ngf, norm_type=norm_type),
                 activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      LayerHelper.get_norm_layer(num_features=ngf * mult * 2, norm_type=norm_type),
                      activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult,
                                  padding_type=padding_type,
                                  norm_type=norm_type,
                                  dropout=dropout if i in dropout_block_list else 0.,
                                  use_bias=use_bias)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias),
                      LayerHelper.get_norm_layer(num_features=int(ngf * mult / 2), norm_type=norm_type),
                      # norm_layer(int(ngf * mult / 2)),
                      activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_type, dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_type, dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_type, dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       LayerHelper.get_norm_layer(num_features=dim, norm_type=norm_type),
                       # norm_layer(dim),
                       nn.ReLU(True),
                       nn.Dropout(dropout)]
        # if use_dropout:
        #     conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       LayerHelper.get_norm_layer(num_features=dim, norm_type=norm_type)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out