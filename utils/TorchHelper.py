#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.

import os
from collections import OrderedDict
import torch
from torch.nn import init
from torch.optim import lr_scheduler
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from typing import Optional
import copy


class TorchHelper:

    @staticmethod
    def get_scheduler(optimizer, n_epoch, scheduler_config: dict):
        policy = scheduler_config["policy"]
        if policy == "linear":
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - n_epoch) / float(scheduler_config["n_epoch_decay"] + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif policy == "cosine_warm":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=scheduler_config["T_0"],
                                                                 T_mult=scheduler_config["T_mult"])
        else:
            raise NotImplementedError(f"Unknown LR policy {policy}")
        return scheduler