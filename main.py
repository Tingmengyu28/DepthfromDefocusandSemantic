import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import time
import os
import random
import json
import h5py
import cv2 as cv
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from networks import LitDDNet


class NYUDepthDataset(Dataset):
    def __init__(self, all_image_id_paths, transform):
        self.transform = transform
        self.all_image_paths = [
            f"nyu2_data/blurred_n2/{image_id_path}"
            for image_id_path in all_image_id_paths
        ]
        self.all_depth_paths = [
            f"nyu2_data/depth/{depth_id_path}"
            for depth_id_path in all_image_id_paths
        ]
        
    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        
        img = cv.imread(self.all_image_paths[index], cv.IMREAD_COLOR)
        img = self.transform(img)
        # depth = torch.tensor(depth, dtype=torch.float32).permute(1, 0).unsqueeze(0)
        depth = cv.imread(self.all_depth_paths[index], cv.IMREAD_GRAYSCALE)
        depth = torch.from_numpy(depth) / 10
        # depth = self.depths[index, :, :]
        # depth = torch.tensor(depth, dtype=torch.float32).permute(1, 0).unsqueeze(0)
        beta = get_sigma_from_depth(depth)

        return img, beta, depth


def all_images_paths(img_path, p):
    return [file for file in os.listdir(img_path) if random.random() <= p]


def get_sigma_from_depth(s2):
    kcam, s1 = args["kcam"], args["s1"]
    if torch.is_tensor(s2):
        s = torch.abs(s1 - s2).div(s2).to(s2.device) * kcam
    elif type(s2) == np.ndarray:
        s = np.divide(np.abs(s1 - s2), s2) * kcam
    else:
        s = np.abs(s1 - s2) / (s2) * kcam
    return s + args["lowest_beta"]


class NYUDepthDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        all_image_paths = all_images_paths("nyu2_data/clean", 1)
        val_paths = all_images_paths("nyu2_data/clean", args["val_ratio"])
        train_paths = [img_id for img_id in all_image_paths if img_id not in val_paths]
        self.train_loader = NYUDepthDataset(train_paths, transform=self.transform)
        self.val_loader = NYUDepthDataset(val_paths, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True, num_workers=args['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, shuffle=False, num_workers=args['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, shuffle=False, num_workers=args['num_workers'])


if __name__ == "__main__":
    with open("configs.json", "r") as f:
        args = json.load(f)
    save_path = os.path.join("outputs/ckpts", time.strftime('%Y-%m-%d-%H-%M-%S'))
    # model = DenseUNet(d_block=BasicBlock, outputSize=(480, 640))
    data_module = NYUDepthDataModule(batch_size=args['batch_size'])
    logger = TensorBoardLogger("outputs/logs", name=time.strftime('%Y-%m-%d-%H-%M-%S'))
    checkpoint_callback = ModelCheckpoint(dirpath=save_path,
                                          monitor="val_loss",
                                          save_last=True,
                                          every_n_epochs=args['save_epoch'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        accelerator='gpu',
        # strategy='ddp',
        devices=args["devices"],
        max_epochs=args['max_epochs'],
        logger=logger,
        precision='32',
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    parser = argparse.ArgumentParser(description='Depth estimation')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    pas = parser.parse_args()

    model = LitDDNet(args=args)
    if pas.mode == 'train':
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
    elif pas.mode == 'test':
        model = LitDDNet.load_from_checkpoint(args['load_path'])
        trainer.test(model, data_module)