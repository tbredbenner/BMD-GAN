#  Copyright (c) 2021 by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch.nn as nn
import torch
import numpy as np
import functools
from typing import Optional


class LayerHelper:
    @staticmethod
    def get_filter(filt_size=3, device=None):
        if device is None:
            device = torch.device("cpu")
        if filt_size == 1:
            a = np.array([1., ])
        elif filt_size == 2:
            a = np.array([1., 1.])
        elif filt_size == 3:
            a = np.array([1., 2., 1.])
        elif filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise NotImplementedError("Not implemented filter size")

        filt = torch.Tensor(a[:, None] * a[None, :], device=device)
        filt = filt / torch.sum(filt)

        return filt

    @staticmethod
    def get_norm_layer(num_features: int, norm_type: Optional[str] = None) -> torch.nn.Module:
        if norm_type is None:
            norm_type = "group"
        if norm_type == "batch":
            return nn.BatchNorm2d(num_features=num_features)
        if norm_type == "sync_batch":
            return nn.SyncBatchNorm(num_features=num_features)
        if norm_type == "group":
            num_group = 32
            while num_features / num_group != num_features // num_group:
                num_group //= 2
            assert num_features / num_group == num_features // num_group, f"{num_features}, {num_group}"
            return nn.GroupNorm(num_groups=num_group, num_channels=num_features)
        if norm_type == "instance":
            return nn.InstanceNorm2d(num_features=num_features)
        if norm_type == "layer":
            return nn.LayerNorm(normalized_shape=num_features)
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
