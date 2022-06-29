#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.

import numpy as np
import torch
from typing import Union


class ImageHelper:

    @staticmethod
    def norm_image(image: Union[np.ndarray, torch.Tensor],
                   image_min: float,
                   image_max: float) -> Union[np.ndarray, torch.Tensor]:
        ret = image
        ret = (ret - image_min) / (image_max - image_min)
        return (ret - 0.5) * 2.

    @staticmethod
    def denorm_image(image: Union[np.ndarray, torch.Tensor],
                     restore_min: float,
                     restore_max: float) -> Union[np.ndarray, torch.Tensor]:
        ret = image
        ret = (ret + 1.) / 2.
        return (ret * (restore_max - restore_min)) + restore_min
