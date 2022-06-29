# Modified from: https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/decode_heads/daformer_head.py

from torch import nn as nn
from torch.nn import functional as F
from ...utils.LayerHelper import LayerHelper
import torch


class DepthwiseSeparableASPPModule(nn.ModuleList):
    def __init__(self,
                 dilations,
                 in_channels,
                 channels,
                 norm_type,
                 padding_type,
                 act_layer):
        super(DepthwiseSeparableASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.norm_type = norm_type
        self.padding_type = padding_type
        for dilation in dilations:
            self.append(
                nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.in_channels,
                                        kernel_size=1 if dilation == 1 else 3,
                                        dilation=dilation,
                                        padding=0 if dilation == 1 else dilation,
                                        padding_mode=padding_type,
                                        groups=self.in_channels),
                              nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.channels,
                                        kernel_size=1),
                              LayerHelper.get_norm_layer(self.channels, norm_type=norm_type),
                              act_layer()))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


def build_layer(in_channels, out_channels, type, **kwargs):

    if type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 dilations,
                 pool,
                 padding_type,
                 norm_type,
                 act_layer,
                 align_corners,
                 # context_cfg=None
                 ):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, channels, 1, padding_mode=padding_type),
                LayerHelper.get_norm_layer(num_features=channels, norm_type=norm_type),
                act_layer()
            )
        else:
            self.image_pool = None
            self.context_layer = None
        ASPP = DepthwiseSeparableASPPModule
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            padding_type=padding_type,
            norm_type=norm_type,
            act_layer=act_layer
        )
        self.bottleneck = nn.Sequential(nn.Conv2d((len(dilations) + int(pool)) * channels,
                                                  channels,
                                                  kernel_size=3,
                                                  padding=1,
                                                  padding_mode=padding_type,
                                                  ),
                                        LayerHelper.get_norm_layer(num_features=channels, norm_type=norm_type),
                                        act_layer())

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x

class BaseDecodeHead(nn.Module):
    """Base class for BaseDecodeHead.
    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 dropout_ratio,
                 in_index=-1,
                 input_transform=None,
                 ignore_index=255,
                 align_corners=False,
        ):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.
        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform
        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.
        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


class DAFormerHead(BaseDecodeHead):

    @staticmethod
    def get_da_former(num_classes,
                      in_channels,
                      dropout_rate,
                      padding_type="reflect",
                      norm_type="group"):
        model = dict(in_channels=in_channels,
                     in_index=[0, 1, 2, 3],
                     channels=256,
                     dropout_ratio=dropout_rate,
                     num_classes=num_classes,
                     align_corners=False,
                     decoder_params=dict(
                         embed_dims=256,
                         embed_cfg=dict(type='mlp'),
                         embed_neck_cfg=dict(type='mlp'),
                         fusion_cfg=dict(
                             type='aspp',
                             dilations=(1, 6, 12, 18),
                             padding_type=padding_type,
                             norm_type=norm_type,
                             act_layer=nn.ReLU,
                             pool=False)))
        return DAFormerHead(**model)

    def __init__(self,
                 in_channels,
                 in_index,
                 channels,
                 dropout_ratio,
                 num_classes,
                 align_corners,
                 decoder_params):
        super(DAFormerHead, self).__init__(input_transform='multiple_select',
                                           in_channels=in_channels,
                                           channels=channels,
                                           num_classes=num_classes,
                                           dropout_ratio=dropout_ratio,
                                           in_index=in_index,
                                           align_corners=align_corners)

        assert not self.align_corners
        decoder_params
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous() \
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)

        return x

import warnings

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)