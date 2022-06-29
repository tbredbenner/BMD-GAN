import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Network.utils.LayerHelper import LayerHelper
from functools import partial

# from lib.models.tools.module_helper import ModuleHelper
# from lib.utils.tools.logger import Logger as Log

if torch.__version__.startswith('1'):
    relu_inplace = True
else:
    relu_inplace = False


def conv3x3(in_planes, out_planes, stride=1, padding_type="reflect"):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, padding_mode=padding_type)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type=None, bn_momentum=0.1,
                 padding_type="reflect"):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding_type=padding_type)
        # self.bn1 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(planes, momentum=bn_momentum)
        self.bn1 = LayerHelper.get_norm_layer(num_features=planes, norm_type=norm_type)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding_type=padding_type)
        # self.bn2 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(planes, momentum=bn_momentum)
        self.bn2 = LayerHelper.get_norm_layer(num_features=planes, norm_type=norm_type)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type=None, bn_momentum=0.1,
                 padding_type="reflect"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(planes, momentum=bn_momentum)
        self.bn1 = LayerHelper.get_norm_layer(num_features=planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, padding_mode=padding_type)
        # self.bn2 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(planes, momentum=bn_momentum)
        self.bn2 = LayerHelper.get_norm_layer(num_features=planes, norm_type=norm_type)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(planes * 4, momentum=bn_momentum)
        self.bn3 = LayerHelper.get_norm_layer(num_features=planes * 4, norm_type=norm_type)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_type="group", bn_momentum=0.1,
                 padding_type="reflect"):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels, norm_type=norm_type, bn_momentum=bn_momentum,
            padding_type=padding_type)
        self.fuse_layers = self._make_fuse_layers(norm_type=norm_type, bn_momentum=bn_momentum,
                                                  padding_type=padding_type)
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            # Log.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            # Log.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            # Log.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1, norm_type=None, bn_momentum=0.1, padding_type="reflect"):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                LayerHelper.get_norm_layer(num_features=num_channels[branch_index] * block.expansion,
                                           norm_type=norm_type)
                # ModuleHelper.BatchNorm2d(bn_type=norm_type)(
                #     num_channels[branch_index] * block.expansion,
                #     momentum=bn_momentum
                # ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                padding_type=padding_type
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    norm_type=norm_type,
                    bn_momentum=bn_momentum,
                    padding_type=padding_type
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, norm_type, bn_momentum=0.1,
                       padding_type="reflect"):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels, norm_type=norm_type, bn_momentum=bn_momentum,
                                      padding_type=padding_type)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, norm_type, bn_momentum=0.1, padding_type="reflect"):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False
                            ),
                            LayerHelper.get_norm_layer(num_features=num_inchannels[i], norm_type=norm_type)
                            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(num_inchannels[i], momentum=bn_momentum),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False, padding_mode=padding_type
                                    ),
                                    LayerHelper.get_norm_layer(num_features=num_outchannels_conv3x3,
                                                               norm_type=norm_type)
                                    # ModuleHelper.BatchNorm2d(bn_type=norm_type)(num_outchannels_conv3x3, momentum=bn_momentum)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False, padding_mode=padding_type
                                    ),
                                    LayerHelper.get_norm_layer(num_features=num_outchannels_conv3x3,
                                                               norm_type=norm_type),
                                    # ModuleHelper.BatchNorm2d(bn_type=norm_type)(num_outchannels_conv3x3, momentum=bn_momentum),
                                    nn.ReLU(inplace=False)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, input_nc, norm_type, padding_type, dropout_rate=0., dropout_layer=nn.Dropout2d):
        self.inplanes = 64
        super(HighResolutionNet, self).__init__()
        dropout_layer = partial(dropout_layer, inplace=False)
        bn_momentum = 0.1
        # if os.environ.get('full_res_stem'):
        #     # Log.info("using full-resolution stem with stride=1")
        #     stem_stride = 1
        #     self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stem_stride, padding=1,
        #                            bias=False, padding_mode=padding_type)
        #     # self.bn1 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(64, momentum=bn_momentum)
        #     self.bn1 = LayerHelper.get_norm_layer(num_features=64, norm_type=norm_type)
        #     self.relu = nn.ReLU(inplace=False)
        #     self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)
        # else:
        stem_stride = 2
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=stem_stride, padding=1,
                               bias=False, padding_mode=padding_type)
        # self.bn1 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(64, momentum=bn_momentum)
        self.bn1 = LayerHelper.get_norm_layer(num_features=64, norm_type=norm_type)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=stem_stride, padding=1,
                               bias=False, padding_mode=padding_type)
        # self.bn2 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(64, momentum=bn_momentum)
        self.bn2 = LayerHelper.get_norm_layer(num_features=64, norm_type=norm_type)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, norm_type=norm_type, bn_momentum=bn_momentum,
                                       padding_type=padding_type)
        self.drop1 = dropout_layer(dropout_rate)

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]

        self.transition1 = self._make_transition_layer([256], num_channels, norm_type=norm_type,
                                                       bn_momentum=bn_momentum, padding_type=padding_type)

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)
        self.drop2 = []
        for _ in range(len(num_channels)):
            self.drop2.append(dropout_layer(dropout_rate))
        self.drop2 = nn.ModuleList(self.drop2)

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)
        self.drop3 = []
        for _ in range(len(num_channels)):
            self.drop3.append(dropout_layer(dropout_rate))
        self.drop3 = nn.ModuleList(self.drop3)

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        self.stage4_n_channels = num_channels.copy()
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, norm_type=norm_type, bn_momentum=bn_momentum,
            padding_type=padding_type)
        self.drop4 = []
        for _ in range(len(num_channels)):
            self.drop4.append(dropout_layer(dropout_rate))
        self.drop4 = nn.ModuleList(self.drop4)

        if os.environ.get('keep_imagenet_head'):
            self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels, norm_type=norm_type, bn_momentum=bn_momentum,
                                               padding_type=padding_type)

    def _make_head(self, pre_stage_channels, norm_type, bn_momentum, padding_type):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Log.info("pre_stage_channels: {}".format(pre_stage_channels))
        # Log.info("head_channels: {}".format(head_channels))

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            norm_type=norm_type,
                                            bn_momentum=bn_momentum,
                                            padding_type=padding_type
                                            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          padding_mode=padding_type),
                # ModuleHelper.BatchNorm2d(bn_type=norm_type)(out_channels, momentum=bn_momentum),
                LayerHelper.get_norm_layer(num_features=out_channels, norm_type=norm_type),
                nn.ReLU(inplace=False)
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(2048, momentum=bn_momentum),
            LayerHelper.get_norm_layer(num_features=2048, norm_type=norm_type),
            nn.ReLU(inplace=False)
        )
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer, norm_type, bn_momentum, padding_type):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                                padding_mode=padding_type
                            ),
                            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(num_channels_cur_layer[i], momentum=bn_momentum),
                            LayerHelper.get_norm_layer(num_features=num_channels_cur_layer[i], norm_type=norm_type),
                            nn.ReLU(inplace=False)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels,
                                outchannels,
                                3,
                                2,
                                1,
                                bias=False,
                                padding_mode=padding_type
                            ),
                            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(outchannels, momentum=bn_momentum),
                            LayerHelper.get_norm_layer(num_features=outchannels, norm_type=norm_type),
                            nn.ReLU(inplace=False)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, norm_type=None, bn_momentum=0.1,
                    padding_type="reflect"):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                LayerHelper.get_norm_layer(num_features=planes * block.expansion, norm_type=norm_type)
                # ModuleHelper.BatchNorm2d(bn_type=norm_type)(planes * block.expansion, momentum=bn_momentum)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_type=norm_type, bn_momentum=bn_momentum,
                            padding_type=padding_type))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(inplanes, planes, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, norm_type=None, bn_momentum=0.1, padding_type="reflect"):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    norm_type,
                    bn_momentum,
                    padding_type=padding_type
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        # if os.environ.get('full_res_stem'):
        #     x = self.conv1(x)
        #     x = self.bn1(x)
        #     x = self.relu(x)
        # else:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.drop1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        for i, (drop_layer, y) in enumerate(zip(self.drop2, y_list)):
            y_list[i] = drop_layer(y)


        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        for i, (drop_layer, y) in enumerate(zip(self.drop3, y_list)):
            y_list[i] = drop_layer(y)

        # if os.environ.get('drop_stage4'):
        #     return y_list

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        for i, (drop_layer, y) in enumerate(zip(self.drop4, y_list)):
            y_list[i] = drop_layer(y)

        # if os.environ.get('keep_imagenet_head'):
        #     # Classification Head
        #     x_list = []
        #     y = self.incre_modules[0](y_list[0])
        #     x_list.append(y)
        #     for i in range(len(self.downsamp_modules)):
        #         y = self.incre_modules[i+1](y_list[i+1]) + \
        #             self.downsamp_modules[i](y)
        #         x_list.append(y)
        #
        #     y = self.final_layer(y)
        #     del x_list[-1]
        #     x_list.append(y)
        #
        #     return x_list

        return y_list


class HighResolutionNext(nn.Module):

    def __init__(self, cfg, input_nc, norm_type, padding_type, **kwargs):
        super(HighResolutionNext, self).__init__()
        # stem net
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=2, padding=1,
                               bias=False, padding_mode=padding_type)
        # self.bn1 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(64)
        self.bn1 = LayerHelper.get_norm_layer(num_features=64, norm_type=norm_type)
        self.relu = nn.ReLU(relu_inplace)

        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition0 = self._make_transition_layer([64], num_channels, norm_type=norm_type,
                                                       padding_type=padding_type)
        self.stage1, pre_stage_channels = self._make_stage(
            self.stage1_cfg, num_channels, norm_type=norm_type)

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_type=norm_type)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, norm_type=norm_type)

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_type=norm_type)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, norm_type=norm_type)

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        self.stage4_n_channels = num_channels.copy()
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_type=norm_type)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, norm_type=norm_type)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer, norm_type, padding_type):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False, padding_mode=padding_type
                            ),
                            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(num_channels_cur_layer[i]),
                            LayerHelper.get_norm_layer(num_features=num_channels_cur_layer[i], norm_type=norm_type),
                            nn.ReLU(relu_inplace)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False, padding_mode=padding_type
                            ),
                            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(outchannels),
                            LayerHelper.get_norm_layer(num_features=outchannels, norm_type=norm_type),
                            nn.ReLU(relu_inplace)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, norm_type=None, padding_type="reflect"):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    norm_type,
                    padding_type=padding_type
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_list = []
        for i in range(self.stage1_cfg['NUM_BRANCHES']):
            if self.transition0[i] is not None:
                x_list.append(self.transition0[i](x))
            else:
                x_list.append(x)
        y_list = self.stage1(x_list)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                if i == 0:
                    x_list.append(self.transition1[i](y_list[0]))
                else:
                    x_list.append(self.transition1[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        return x


class HighResolutionNetPlus(nn.Module):

    def __init__(self, cfg, input_nc, norm_type, padding_type, **kwargs):
        self.inplanes = 64
        super(HighResolutionNetPlus, self).__init__()
        bn_momentum = 0.1
        stem_stride = 2
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=stem_stride, padding=1,
                               bias=False, padding_mode=padding_type)
        # self.bn1 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(64, momentum=bn_momentum)
        self.bn1 = LayerHelper.get_norm_layer(num_features=64, norm_type=norm_type)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=stem_stride, padding=1,
                               bias=False, padding_mode=padding_type)
        # self.bn2 = ModuleHelper.BatchNorm2d(bn_type=norm_type)(64, momentum=bn_momentum)
        self.bn2 = LayerHelper.get_norm_layer(num_features=64, norm_type=norm_type)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, norm_type=norm_type, bn_momentum=bn_momentum,
                                       padding_type=padding_type)

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]

        self.transition1 = self._make_transition_layer([256], num_channels, norm_type=norm_type,
                                                       bn_momentum=bn_momentum, padding_type=padding_type)

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)

        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        self.stage4_n_channels = num_channels.copy()
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, norm_type=norm_type, bn_momentum=bn_momentum,
            padding_type=padding_type)

        num_channels = num_channels[: -1]
        self.stage3_2, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)

        num_channels = num_channels[: -1]
        self.stage2_2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)
        self.layer1_2 = nn.Sequential(self._make_layer(Bottleneck, num_channels[0], num_channels[0], 4, norm_type=norm_type,
                                                       bn_momentum=bn_momentum, padding_type=padding_type),
                                      nn.Conv2d(num_channels[0] * Bottleneck.expansion,
                                                num_channels[0],
                                                3,
                                                1,
                                                1,
                                                bias=False,
                                                padding_mode=padding_type
                                                ),
                                      LayerHelper.get_norm_layer(num_features=num_channels[0], norm_type=norm_type),
                                      nn.ReLU(inplace=False)
                                      )
        # # LayerHelper.get_norm_layer(num_features=num_channels[0], norm_type=norm_type),
        # nn.ReLU(inplace=False)]
        # self.layer1_2 = nn.Sequential(*layer1_2)
        # layer_config = dict(NUM_MODULES=1,
        #                     NUM_BRANCHES=1,
        #                     NUM_BLOCKS=[4],
        #                     NUM_CHANNELS=[self.stage2_cfg["NUM_CHANNELS"][0]],
        #                     BLOCK="BASIC",
        #                     FUSE_METHOD=self.stage2_cfg["FUSE_METHOD"])
        # num_channels = num_channels[: -1]
        # self.stage1_2, pre_stage_channels = self._make_stage(
        #     layer_config, num_channels, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type)

    def _make_head(self, pre_stage_channels, norm_type, bn_momentum, padding_type):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Log.info("pre_stage_channels: {}".format(pre_stage_channels))
        # Log.info("head_channels: {}".format(head_channels))

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            norm_type=norm_type,
                                            bn_momentum=bn_momentum,
                                            padding_type=padding_type
                                            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          padding_mode=padding_type),
                # ModuleHelper.BatchNorm2d(bn_type=norm_type)(out_channels, momentum=bn_momentum),
                LayerHelper.get_norm_layer(num_features=out_channels, norm_type=norm_type),
                nn.ReLU(inplace=False)
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(2048, momentum=bn_momentum),
            LayerHelper.get_norm_layer(num_features=2048, norm_type=norm_type),
            nn.ReLU(inplace=False)
        )
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer, norm_type, bn_momentum, padding_type):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                                padding_mode=padding_type
                            ),
                            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(num_channels_cur_layer[i], momentum=bn_momentum),
                            LayerHelper.get_norm_layer(num_features=num_channels_cur_layer[i], norm_type=norm_type),
                            nn.ReLU(inplace=False)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels,
                                outchannels,
                                3,
                                2,
                                1,
                                bias=False,
                                padding_mode=padding_type
                            ),
                            # ModuleHelper.BatchNorm2d(bn_type=norm_type)(outchannels, momentum=bn_momentum),
                            LayerHelper.get_norm_layer(num_features=outchannels, norm_type=norm_type),
                            nn.ReLU(inplace=False)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, norm_type=None, bn_momentum=0.1,
                    padding_type="reflect"):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                LayerHelper.get_norm_layer(num_features=planes * block.expansion, norm_type=norm_type)
                # ModuleHelper.BatchNorm2d(bn_type=norm_type)(planes * block.expansion, momentum=bn_momentum)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_type=norm_type, bn_momentum=bn_momentum,
                            padding_type=padding_type))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(inplanes, planes, norm_type=norm_type, bn_momentum=bn_momentum, padding_type=padding_type))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, norm_type=None, bn_momentum=0.1, padding_type="reflect"):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    norm_type,
                    bn_momentum,
                    padding_type=padding_type
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        # if os.environ.get('full_res_stem'):
        #     x = self.conv1(x)
        #     x = self.bn1(x)
        #     x = self.relu(x)
        # else:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        i = len(y_list) - 1
        x_list = self.stage3_2(y_list[: i])
        for j in range(len(x_list)):
            y_list[j] = x_list[j]

        i -= 1
        x_list = self.stage2_2(y_list[: i])
        for j in range(len(x_list)):
            y_list[j] = x_list[j]

        y_list[0] = self.layer1_2(y_list[0])
        return y_list


def get_hrnet_backbone(backbone, input_nc, norm_type, padding_type, dropout_rate, dropout_layer):
    from .config import MODEL_CONFIGS
    if backbone in ["hrnet18", "hrnet32", "hrnet48", "hrnet64"]:
        return HighResolutionNet(MODEL_CONFIGS[backbone],
                                 input_nc=input_nc,
                                 norm_type=norm_type,
                                 padding_type=padding_type,
                                 dropout_rate=dropout_rate,
                                 dropout_layer=dropout_layer
                                 )
    if backbone in ["hrnet18+", "hrnet32+", "hrnet48+", "hrnet64+"]:
        raise NotImplementedError
        return HighResolutionNetPlus(MODEL_CONFIGS[backbone], input_nc=input_nc, norm_type=norm_type,
                                     padding_type=padding_type)
    elif backbone == "hrnet2x20":
        raise NotImplementedError
        return HighResolutionNext(MODEL_CONFIGS[backbone], input_nc=input_nc, norm_type=norm_type,
                                  padding_type=padding_type)
    else:
        raise NotImplementedError("Unknown backbone {}.".format(backbone))

# class HRNetBackbone(object):
#     def __init__(self, backbone, ):
#         self.__backbone = backbone
#
#     def __call__(self):
#         arch = self.configer.get('network', 'backbone')
#         from .config import MODEL_CONFIGS
#
#         if arch == 'hrnet18':
#             arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet18'],
#                                          norm_type='inplace_abn',
#                                          bn_momentum=0.1)
#             # arch_net = ModuleHelper.load_model(arch_net,
#             #                                    pretrained=self.configer.get('network', 'pretrained'),
#             #                                    all_match=False,
#             #                                    network='hrnet')
#
#         elif arch == 'hrnet32':
#             arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet32'],
#                                          norm_type='inplace_abn',
#                                          bn_momentum=0.1)
#             # arch_net = ModuleHelper.load_model(arch_net,
#             #                                    pretrained=self.configer.get('network', 'pretrained'),
#             #                                    all_match=False,
#             #                                    network='hrnet')
#
#         elif arch == 'hrnet48':
#             arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet48'],
#                                          norm_type='inplace_abn',
#                                          bn_momentum=0.1)
#             # arch_net = ModuleHelper.load_model(arch_net,
#             #                                    pretrained=self.configer.get('network', 'pretrained'),
#             #                                    all_match=False,
#             #                                    network='hrnet')
#
#         elif arch == 'hrnet64':
#             arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet64'],
#                                          norm_type='inplace_abn',
#                                          bn_momentum=0.1)
#             # arch_net = ModuleHelper.load_model(arch_net,
#             #                                    pretrained=self.configer.get('network', 'pretrained'),
#             #                                    all_match=False,
#             #                                    network='hrnet')
#
#         elif arch == 'hrnet2x20':
#             arch_net = HighResolutionNext(MODEL_CONFIGS['hrnet2x20'],
#                                           norm_type=self.configer.get('network', 'bn_type'))
#             # arch_net = ModuleHelper.load_model(arch_net,
#             #                                    pretrained=self.configer.get('network', 'pretrained'),
#             #                                    all_match=False,
#             #                                    network='hrnet')
#
#         else:
#             raise Exception('Architecture undefined!')
#
#         return arch_net
