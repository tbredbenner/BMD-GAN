#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from .Generator import *
from .Discriminator import *
import torch.nn as nn
from torch.nn import init


def get_generator(config: str,
                  input_nc: int,
                  output_nc: int,
                  img_dsize: tuple[int, int],
                  norm_type="group",
                  padding_type="reflect",
                  init_type="normal",
                  init_gain=0.02) -> nn.Module:
    if config in ResnetGenerator.config_names():
        net = ResnetGenerator
    elif config in DAFormerGenerator.config_names():
        net = DAFormerGenerator
    elif config in HRNetGenrator.config_names():
        net = HRNetGenrator
    elif config in HRTGenerator.config_names():
        net = HRTGenerator
    else:
        raise RuntimeError(f"Unknown config {config}")
    net = net(config=config,
              input_nc=input_nc,
              output_nc=output_nc,
              img_dsize=img_dsize,
              norm_type=norm_type,
              padding_type=padding_type)
    _init_weights(net, init_type, init_gain)
    return net


def get_discriminator(input_nc, norm_type="group", init_type="normal", init_gain=0.02) -> MultiscaleDiscriminator:
    net = MultiscaleDiscriminator(input_nc=input_nc, norm_type=norm_type)
    _init_weights(net, init_type, init_gain)
    return net


def _init_weights(net: nn.Module, init_type='normal', init_gain=0.02) -> None:

    """
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func
