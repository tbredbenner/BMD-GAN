# Obtained from: https://github.com/NVlabs/SegFormer
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------

#  Modified from: https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/backbones/mix_transformer.py


from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn


class DWConv(nn.Module):

    def __init__(self, dim=768, padding_type="reflect"):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim, padding_mode=padding_type)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 padding_type="reflect"):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, padding_type=padding_type)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 padding_type="reflect"):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio, padding_mode=padding_type)
            # self.norm = LayerHelper.get_norm_layer(num_features=dim, norm_type=norm_type)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 sr_ratio=1,
                 padding_type="reflect",
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # self.norm1 = LayerHelper.get_norm_layer(num_features=dim, norm_type=norm_type)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            padding_type=padding_type)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        # self.norm2 = LayerHelper.get_norm_layer(num_features=dim, norm_type=norm_type)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            padding_type=padding_type)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_dsize,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768,
                 padding_type="reflect",
                 ):
        super().__init__()
        # img_size = to_2tuple(img_size)
        img_size = img_dsize[::-1]
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
            padding_mode=padding_type)
        self.norm = nn.LayerNorm(embed_dim)
        # self.norm = LayerHelper.get_norm_layer(num_features=embed_dim, norm_type=norm_type)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    configs = {
        "mit_b0": {"patch_size": 4,
                   "embed_dims": [32, 64, 160, 256],
                   "num_heads": [1, 2, 5, 8],
                   "mlp_ratios": [4, 4, 4, 4],
                   "qkv_bias": True,
                   # "norm_layer": nn.LayerNorm,
                   "depths": [2, 2, 2, 2],
                   "sr_ratios": [8, 4, 2, 1]},
        "mit_b1": {"patch_size": 4,
                   "embed_dims": [64, 128, 320, 512],
                   "num_heads": [1, 2, 5, 8],
                   "mlp_ratios": [4, 4, 4, 4],
                   "qkv_bias": True,
                   # "norm_layer": nn.LayerNorm,
                   "depths": [2, 2, 2, 2],
                   "sr_ratios": [8, 4, 2, 1]},
        "mit_b2": {"patch_size": 4,
                   "embed_dims": [64, 128, 320, 512],
                   "num_heads": [1, 2, 5, 8],
                   "mlp_ratios": [4, 4, 4, 4],
                   "qkv_bias": True,
                   # "norm_layer": nn.LayerNorm,
                   "depths": [3, 4, 6, 3],
                   "sr_ratios": [8, 4, 2, 1]},
        "mit_b3": {"patch_size": 4,
                   "embed_dims": [64, 128, 320, 512],
                   "num_heads": [1, 2, 5, 8],
                   "mlp_ratios": [4, 4, 4, 4],
                   "qkv_bias": True,
                   # "norm_layer": nn.LayerNorm,
                   "depths": [3, 4, 18, 3],
                   "sr_ratios": [8, 4, 2, 1]},
        "mit_b4": {"patch_size": 4,
                   "embed_dims": [64, 128, 320, 512],
                   "num_heads": [1, 2, 5, 8],
                   "mlp_ratios": [4, 4, 4, 4],
                   "qkv_bias": True,
                   # "norm_layer": nn.LayerNorm,
                   "depths": [3, 8, 27, 3],
                   "sr_ratios": [8, 4, 2, 1]},
        "mit_b5": {"patch_size": 4,
                   "embed_dims": [64, 128, 320, 512],
                   "num_heads": [1, 2, 5, 8],
                   "mlp_ratios": [4, 4, 4, 4],
                   "qkv_bias": True,
                   # "norm_layer": nn.LayerNorm,
                   "depths": [3, 6, 40, 3],
                   "sr_ratios": [8, 4, 2, 1]}
    }

    @staticmethod
    def get_model(config: str,
                  img_dsize,
                  input_nc,
                  dropout_rate,
                  padding_type="reflect",
                  ):
        model = partial(MixVisionTransformer,
                        img_dsize=img_dsize,
                        in_chans=input_nc,
                        dropout_rate=dropout_rate,
                        padding_type=padding_type)
        return model(**(MixVisionTransformer.configs[config]))

    def __init__(self,
                 # img_size,
                 img_dsize,
                 in_chans,
                 # num_classes,
                 patch_size=16,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 dropout_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 freeze_patch_embed=False,
                 padding_type="reflect",
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        #
        # assert not (init_cfg and pretrained), \
        #     'init_cfg and pretrained cannot be setting at the same time'
        # if isinstance(pretrained, str) or pretrained is None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated, '
        #                   'please use "init_cfg" instead')
        # else:
        #     raise TypeError('pretrained must be a str or None')

        # self.num_classes = num_classes
        self.depths = depths

        # self.pretrained = pretrained
        # self.init_cfg = init_cfg

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_dsize=img_dsize,
            # img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            padding_type=padding_type,
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_dsize=(img_dsize[0] // 4, img_dsize[1] // 4),
            # img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
            padding_type=padding_type,
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_dsize=(img_dsize[0] // 8, img_dsize[1] // 8),
            # img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
            padding_type=padding_type,
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_dsize=(img_dsize[0] // 16, img_dsize[1] // 16),
            # img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
            padding_type=padding_type,
        )
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                padding_type=padding_type,
                sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])
        # self.norm1 = LayerHelper.get_norm_layer(num_features=embed_dims[0], norm_type=norm_type)

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                padding_type=padding_type,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])
        # self.norm2 = LayerHelper.get_norm_layer(num_features=embed_dims[1], norm_type=norm_type)

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                padding_type=padding_type,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])
        # self.norm3 = LayerHelper.get_norm_layer(num_features=embed_dims[2], norm_type=norm_type)

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                padding_type=padding_type,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])
        # self.norm4 = LayerHelper.get_norm_layer(num_features=embed_dims[3], norm_type=norm_type)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(
    #         self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x