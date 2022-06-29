#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader
from utils.OSHelper import OSHelper
from Network import network
from Dataset import DefaultDataset
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import functools
from utils.TorchHelper import TorchHelper
import torch
from tqdm import tqdm
from ImagePool import ImagePool
from Network.Loss import *
from utils.ImageHelper import ImageHelper
from datetime import datetime
import random
from torch.cuda.amp import autocast


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(False)


def main():
    with open("config/default.yaml", 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    output_dir = OSHelper.path_join("runs", config["name"], "train")
    OSHelper.mkdirs(output_dir)
    device = torch.device(config["gpu_id"])

    netG = network.get_generator(**config["model_config"]["generator"]).to(device)
    netD = network.get_discriminator(**config["model_config"]["discriminator"]).to(device)

    dataset_config = config["dataset_config"]
    default_dataset = functools.partial(DefaultDataset,
                                        A_space_norms=dataset_config["A_space_norms"],
                                        B_space_norms=dataset_config["B_space_norms"],
                                        load_dsize=dataset_config["load_dsize"],
                                        re_dsize=dataset_config["re_dsize"])

    training_dataset = default_dataset(A_space_data_root=dataset_config["train_A_space_data_root"],
                                       B_space_data_root=dataset_config["train_B_space_data_root"],
                                       aug_config=dataset_config["aug_config"])

    test_dataset = default_dataset(A_space_data_root=dataset_config["test_A_space_data_root"],
                                   aug_config="none")
    g = torch.Generator()
    g.manual_seed(config["seed"])
    training_dataloader = DataLoader(dataset=training_dataset,
                                     batch_size=config["training_config"]["batch_size"],
                                     num_workers=config["training_config"]["num_worker"],
                                     shuffle=True,
                                     worker_init_fn=seed_worker,
                                     generator=g,
                                     pin_memory=True)
    g = torch.Generator()
    g.manual_seed(config["seed"])
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config["training_config"]["batch_size"],
                                 num_workers=config["training_config"]["num_worker"],
                                 shuffle=True,
                                 worker_init_fn=seed_worker,
                                 generator=g,
                                 pin_memory=True)

    training_config = config["training_config"]

    if training_config["optimizer"] == "adam":
        optimizer_class = Adam
    else:
        assert training_config["optimizer"] == "adamw"
        optimizer_class = AdamW

    optimizer_class = functools.partial(optimizer_class,
                                        lr=training_config["init_lr"],
                                        betas=training_config["betas"],
                                        weight_decay=training_config["weight_decay"])
    G_optimizer = optimizer_class(params=netG.parameters())
    D_optimizer = optimizer_class(params=netD.parameters())

    get_scheduler = functools.partial(TorchHelper.get_scheduler,
                                      n_epoch=training_config["n_epoch"],
                                      scheduler_config=training_config["lr_scheduler_config"])
    G_schedular = get_scheduler(optimizer=G_optimizer)
    D_schedular = get_scheduler(optimizer=D_optimizer)

    G_grad_scaler = torch.cuda.amp.GradScaler()
    D_grad_scaler = torch.cuda.amp.GradScaler()

    image_pool = ImagePool(10)

    crit_AE = torch.nn.L1Loss()
    crit_GAN = GANLoss().to(device)
    crit_GC = GradientCorrelationLoss2D(grad_method="sobel").to(device)

    lambda_GAN = 1.
    lambda_GC = 1.
    lambda_FM = 10.
    lambda_AE = 100.

    n_epoch = training_config["n_epoch"] + 1
    if training_config["lr_scheduler_config"]["policy"] == "linear":
        n_epoch += training_config["lr_scheduler_config"]["n_epoch_decay"]
    for epoch in range(1, n_epoch):
        print(f"Epoch {epoch} ({datetime.now()}), lr {G_schedular.get_last_lr()}")
        epoch_losses = {k: 0. for k in ["G_GAN", "G_GC", "G_FM", "G_AE", "D_fake", "D_real"]}
        for data in tqdm(training_dataloader,
                         total=len(training_dataloader),
                         desc="Train"):
            with autocast():
                real_A = data["A_image"].to(device)
                real_B = data["B_image"].to(device)
                fake_B = netG(real_A)
                D_pred_fake_pool = netD(image_pool.query(torch.cat([real_A, fake_B], dim=1).detach()))
                D_pred_real = netD(torch.cat([real_A, real_B], dim=1))
                D_pred_fake = netD(torch.cat([real_A, fake_B], dim=1))
                step_losses = {"G_GAN": crit_GAN(D_pred_fake, True) * lambda_GAN,
                               "G_FM": _calc_FM_loss(D_pred_fake,
                                                     D_pred_real,
                                                     netD.n_layer,
                                                     netD.num_D,
                                                     crit_AE) * lambda_FM,
                               "G_AE": crit_AE(real_B, fake_B) * lambda_AE,
                               "D_fake": crit_GAN(D_pred_fake_pool, False) * 0.5,
                               "D_real": crit_GAN(D_pred_real, True) * 0.5,
                               "G_GC": 0}
                B = real_A.shape[0]
                for i in range(B):
                    step_losses["G_GC"] += crit_GC(real_B[:, i: i + 1], fake_B[:, i: i + 1])
                step_losses["G_GC"] += lambda_GC

            G_sum = 0.
            D_sum = 0.
            for k, v in step_losses.items():
                if k.startswith("G"):
                    G_sum += v
                else:
                    D_sum += v
                epoch_losses[k] += v.detach().cpu().numpy()

            G_optimizer.zero_grad()
            G_grad_scaler.scale(G_sum).backward()
            G_grad_scaler.step(G_optimizer)
            G_grad_scaler.update()

            D_optimizer.zero_grad()
            D_grad_scaler.scale(D_sum).backward()
            D_grad_scaler.step(D_optimizer)
            D_grad_scaler.update()
        msg = ""
        for k, v in epoch_losses.items():
            epoch_losses[k] = v / len(training_dataloader)
            msg += "%s: %.3f " % (k, epoch_losses[k])
        print(msg)


        with torch.no_grad():
            test_output_dir = OSHelper.path_join(output_dir, "image")
            OSHelper.mkdirs(test_output_dir)
            for data in test_dataloader:
                real_As = data["A_image"].to(device)
                fake_Bs = netG(real_As)

                real_As = ImageHelper.denorm_image(real_As, 0., 255.).detach().cpu().numpy()  # [0, 255.] (B,1,H,W)
                fake_Bs = ImageHelper.denorm_image(fake_Bs, 0., 255.).detach().cpu().numpy()  # [0, 255.] (B,3,H,W)
                real_As = np.transpose(real_As, (0, 2, 3, 1))
                fake_Bs = np.transpose(fake_Bs, (0, 2, 3, 1))
                real_As = np.round(real_As).astype(np.uint8)
                fake_Bs = np.round(fake_Bs).astype(np.uint8)
                real_As = np.squeeze(real_As)
                B = 1
                for i in range(B):
                    real_A = real_As[i]  # (0, 255) (H, W)
                    real_A = cv2.applyColorMap(real_A, cv2.COLORMAP_JET)  # (H, W, 3
                    fake_B = fake_Bs[i]  # (0, 255) (H, W, 3)

                    cv2.imwrite(OSHelper.path_join(test_output_dir, f"e{epoch}_real_A.png"), real_A)
                    cv2.imwrite(OSHelper.path_join(test_output_dir, f"e{epoch}_fake_B.png"), fake_B)

                    pass
                break

        G_schedular.step()
        D_schedular.step()


def _calc_FM_loss(pred_fake: torch.Tensor,
                  pred_real: torch.Tensor,
                  n_layers_D: int,
                  num_D: int,
                  criterion: torch.nn.Module):
    loss_G_FM = 0.
    feat_weights = 4. / (n_layers_D + 1)
    D_weights = 1. / num_D
    for i in range(num_D):
        for j in range(len(pred_fake[i]) - 1):
            loss_G_FM += D_weights * feat_weights * criterion(pred_fake[i][j], pred_real[i][j].detach())
    return loss_G_FM
    pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    main()
