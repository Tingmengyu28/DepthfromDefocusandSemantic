import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import math
from torchmetrics.functional import structural_similarity_index_measure as ssim
from d3networks.dense_decoders_multitask_auto import DenseUNet
from d3networks.conv_blocks import BasicBlock
from dmenet import VGG19_down
from twohdednet import TwoHeadedDepthDeblurNet
from loss import *


def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = F.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad

def tv_norm(image, tv_weight):
    if len(image.shape) == 4:
        diff_i = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        diff_j = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    elif len(image.shape) == 3:
        diff_i = torch.abs(image[:, :, 1:] - image[:, :, :-1])
        diff_j = torch.abs(image[:, 1:, :] - image[:, :-1, :])
    tv_norm = torch.mean(diff_i) + torch.mean(diff_j)

    return tv_weight * tv_norm


def CharbonnierLoss(images, output_images, weight):
    diff = images - output_images
    loss = torch.mean(torch.sqrt(diff ** 2 + 10 ** -6))
    return loss * weight


def cal_AbsRel(depths, output_depths):
    return torch.mean(torch.abs(depths - output_depths) / depths)


def cal_delta(depths, output_depths, p):
    delta1 = output_depths / depths
    delta2 = depths / output_depths
    delta = torch.max(delta1, delta2)
    threshold = 1.25 ** p
    delta_bi = torch.zeros_like(delta, requires_grad=False, device=delta.device)
    delta_bi[delta < threshold] = 1
    return torch.mean(delta_bi)


def diff_penalty(W, d, S, lambda_p):
    return torch.mean(W * (d - S)) * lambda_p


def denormalize(image, mean, std):
    mean = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return image * std + mean


def visualize_sample(image, beta, predicted_beta, depth, predicted_depth, logger, step):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = denormalize(image, mean, std)

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    # 原图
    axes[0].imshow(image)
    axes[0].set_title("Original Coded Image")
    axes[0].axis("off")

    # 深度图标签
    axes[1].imshow(beta.squeeze().cpu().numpy(), cmap="plasma")
    axes[1].set_title("Ground Truth Blur")
    axes[1].axis("off")

    # 预测深度图
    axes[2].imshow(predicted_beta.squeeze().cpu().detach().numpy(), cmap="plasma")
    axes[2].set_title("Predicted Blur")
    axes[2].axis("off")

    # 深度图标签
    axes[3].imshow(depth.squeeze().cpu().numpy(), cmap="plasma")
    axes[3].set_title("Ground Truth Depth")
    axes[3].axis("off")

    # 预测深度图
    axes[4].imshow(predicted_depth.squeeze().cpu().detach().numpy(), cmap="plasma")
    axes[4].set_title("Predicted Depth")
    axes[4].axis("off")

    plt.savefig("test.png")
    logger.experiment.add_figure("Sample Visualization", fig, global_step=step)
    plt.close(fig)


class UNet(nn.Module):
    def __init__(self, in_chn=1, out_chn=1):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_chn, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_chn, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(kernel_size=2, stride=2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(kernel_size=2, stride=2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(kernel_size=2, stride=2)(enc3))

        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2, stride=2)(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


class SelfAttention(nn.Module):
    def __init__(self, dim, window_size):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        pad = self.window_size // 2
        q = F.pad(q, (pad, pad, pad, pad), mode='constant', value=0)
        k = F.pad(k, (pad, pad, pad, pad), mode='constant', value=0)
        v = F.pad(v, (pad, pad, pad, pad), mode='constant', value=0)

        windows_q = q.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        windows_k = k.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        windows_v = v.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)

        windows_q = windows_q.contiguous().view(B, C, H, W, self.window_size * self.window_size)
        windows_k = windows_k.contiguous().view(B, C, H, W, self.window_size * self.window_size)
        windows_v = windows_v.contiguous().view(B, C, H, W, self.window_size * self.window_size)

        attn = torch.einsum('bchwp,bchkp->bhwk', [windows_q, windows_k])
        attn = attn.softmax(dim=-1)

        weighted_v = torch.einsum('bhwk,bchkp->bchwp', [attn, windows_v])
        weighted_v = weighted_v.sum(dim=-1)

        return weighted_v


class DepthPredictor(pl.LightningModule):
    def __init__(self, window_size):
        super(DepthPredictor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.attention = SelfAttention(dim=128, window_size=window_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        return x.squeeze(1)
    

class AttCnn(nn.Module):
    def __init__(self, nf=64, extra_chn=3):
        super(AttCnn, self).__init__()
        self.extra_chn = extra_chn
        if extra_chn > 0:
            self.sft1 = AttLayer(nf, extra_chn)

        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, extra_maps):
        mul1, add1 = self.sft1(extra_maps) if self.extra_chn > 0 else (1, 0)
        mul_feature_maps = feature_maps * mul1
        out = add1 / (1 + mul_feature_maps)
        return out


class AttLayer(nn.Module):
    def __init__(self, out_chn=64, extra_chn=3):
        super(AttLayer, self).__init__()

        nf1 = out_chn // 8
        nf2 = out_chn // 4

        self.conv1 = nn.Conv2d(extra_chn, nf1, kernel_size=1, stride=1, padding=0)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf1, nf2, kernel_size=1, stride=1, padding=0)
        self.leaky2 = nn.LeakyReLU(0.2)
        self.mul_conv = nn.Conv2d(nf2, out_chn, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()
        self.add_conv = nn.Conv2d(nf2, out_chn, kernel_size=1, stride=1, padding=0)

    def forward(self, extra_maps):
        fea1= self.leaky1(self.conv1(extra_maps))
        fea2= self.leaky2(self.conv2(fea1))
        mul = self.mul_conv(fea2)
        add = self.add_conv(fea2)
        return mul, add

class AttResBlock(nn.Module):
    def __init__(self, nf=64, extra_chn=1):
        super(AttResBlock, self).__init__()
        self.extra_chn = extra_chn
        if extra_chn > 0:
            self.sft1 = AttLayer(nf, extra_chn)
            self.sft2 = AttLayer(nf, extra_chn)

        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, feature_maps, extra_maps):
        '''
        Input:
            feature_maps: N x c x h x w
            extra_maps: N x c x h x w or None
        '''
        mul1, add1 = self.sft1(extra_maps) if self.extra_chn > 0 else (1, 0)
        fea1 = self.conv1(self.lrelu1(feature_maps * mul1 + add1))

        mul2, add2 = self.sft2(extra_maps) if self.extra_chn > 0 else (1, 0)
        fea2 = self.conv2(self.lrelu2(fea1 * mul2 + add2))
        out = torch.add(feature_maps, fea2)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_chn=64, out_chn=128, extra_chn=4, n_resblocks=1, downsample=True):
        super(DownBlock, self).__init__()
        self.body = nn.ModuleList([AttResBlock(in_chn, extra_chn) for ii in range(n_resblocks)])
        if downsample:
            self.downsampler = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=2, padding=1)
        else:
            self.downsampler = nn.Identity()

    def forward(self, x, extra_maps):
        for op in self.body:
            x= op(x, extra_maps)
        out =self.downsampler(x)
        return out, x

class UpBlock(nn.Module):
    def __init__(self, in_chn=128, out_chn=64, n_resblocks=1):
        super(UpBlock, self).__init__()
        self.upsampler = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=2, stride=2, padding=0)
        self.body = nn.ModuleList([AttResBlock(nf=out_chn, extra_chn=0) for _ in range(n_resblocks)])

    def forward(self, x, bridge):
        x_up = self.upsampler(x)
        for ii, op in enumerate(self.body):
            if ii == 0:
                x_up = op(x_up+bridge, None)
            else:
                x_up = op(x_up, None)
        return x_up

class AttResUNet(nn.Module):
    def __init__(self, in_chn=3,
                 extra_chn=1,
                 out_chn=1,
                 n_resblocks=2,
                 n_feat=[64, 128, 196, 256],
                 extra_mode='Input'):
        """
        Args:
            in_chn: number of input channels
            extra_chn: number of other channels, e.g., noise variance, kernel information
            out_chn: number of output channels.
            n_resblocks: number of resblocks in each scale of UNet
            n_feat: number of channels in each scale of UNet
            extra_mode: Null, Input, Down or Both
        """
        super(AttResUNet, self).__init__()

        self.out_chn = out_chn

        assert isinstance(n_feat, tuple) or isinstance(n_feat, list)
        self.depth = len(n_feat)

        self.extra_mode = extra_mode.lower()
        assert self.extra_mode in ['null', 'input', 'down', 'both']

        if self.extra_mode in ['down', 'null']:
            self.head = nn.Conv2d(in_chn, n_feat[0], kernel_size=3, stride=1, padding=1)
        else:
            self.head = nn.Conv2d(in_chn+extra_chn, n_feat[0], kernel_size=3, stride=1, padding=1)

        extra_chn_down = extra_chn if self.extra_mode.lower() in ['down', 'both'] else 0
        self.down_path = nn.ModuleList()
        for ii in range(self.depth):
            if ii+1 < self.depth:
                self.down_path.append(DownBlock(n_feat[ii], n_feat[ii+1],
                                                extra_chn=extra_chn_down,
                                                n_resblocks=n_resblocks,
                                                downsample=True))
            else:
                self.down_path.append(DownBlock(n_feat[ii], n_feat[ii],
                                      extra_chn=extra_chn_down,
                                      n_resblocks=n_resblocks,
                                      downsample=False))

        self.up_path = nn.ModuleList()
        for jj in reversed(range(self.depth - 1)):
            self.up_path.append(UpBlock(n_feat[jj+1], n_feat[jj], n_resblocks))

        self.tail = nn.Conv2d(n_feat[0], out_chn, kernel_size=3, stride=1, padding=1)

    def forward(self, x_in, extra_maps_in):
        '''
        Input:
            x_in: N x [] x h x w
            extra_maps: N x []
        '''
        h, w = x_in.shape[-2:]
        x = pad_input(x_in, 2**(self.depth-1))
        if not self.extra_mode == 'null':
            extra_maps = pad_input(extra_maps_in, 2**(self.depth-1))

        if self.extra_mode in ['input', 'both']:
            x = self.head(torch.cat([x, extra_maps], 1))
        else:
            x = self.head(x)

        blocks = []
        if self.extra_mode in ['down', 'both']:
            extra_maps_down = [extra_maps,]
        for ii, down in enumerate(self.down_path):
            if self.extra_mode in ['down', 'both']:
                x, before_down = down(x, extra_maps_down[ii])
            else:
                x, before_down = down(x, None)
            if ii != len(self.down_path)-1:
                blocks.append(before_down)
                if self.extra_mode in ['down', 'both']:
                    extra_maps_down.append(F.interpolate(extra_maps, x.shape[-2:], mode='nearest'))

        for jj, up in enumerate(self.up_path):
            x = up(x, blocks[-jj-1])

        if self.out_chn < x_in.shape[1]:
            out = self.tail(x)[..., :h, :w]
        else:
            out = self.tail(x)[..., :h, :w] + x_in

        return out
    

class LitDDNet(pl.LightningModule):
    def __init__(self, args):
        super(LitDDNet, self).__init__()
        self.save_hyperparameters()
        self.alpha = args['alpha']
        self.lambda0 = args['lambda0']
        self.lambda1 = args['lambda1']
        self.mu = args['mu']
        self.learning_rate = args['learning_rate']
        self.min_learning_rate = args['min_learning_rate']
        self.num_epochs = args['max_epochs']
        self.focal = args['s1']
        self.kcam = args['kcam']
        self.model_name = args['model_name']
        self.criterion = nn.MSELoss()
        self.L1Loss = nn.L1Loss()

        # self.knet = UNet()
        if self.model_name == 'KDNet':
            self.knet = DenseUNet(BasicBlock)
            self.dnet = AttResUNet(in_chn=1, extra_chn=3)
        elif self.model_name == '2HDEDNet':
            self.net = TwoHeadedDepthDeblurNet()
        elif self.model_name == 'D3Net':
            self.d3net = DenseUNet(BasicBlock)
        elif self.model_name == 'KWANet':
            self.knet = DenseUNet(BasicBlock)
            self.wnet = AttResUNet(in_chn=1, extra_chn=3)
        # self.d3net = DenseUNet(BasicBlock)
        # self.dnet = DepthPredictor(args['window_size'])
        # for para in self.dnet.parameters():
        #     para.requires_grad = False

    def forward(self, x):
        if self.model_name == 'KDNet':
            out = self.knet(x)
            return out, self.dnet(out, x)
        elif self.model_name == '2HDEDNet':
            depth, aif = self.net(x)
            return depth, aif
        elif self.model_name == 'D3Net':
            out = self.d3net(x)
            return out, out
        elif self.model_name == 'KWANet':
            out = self.knet(x)
            w = self.wnet(out, x)
            return out, w * out + 1 / self.focal, w


    def training_step(self, batch, batch_idx):
        images, betas, depths = batch
        if self.model_name == 'KDNet':
            output_betas, output_depths = self(images)
            output_betas, output_depths = output_betas.squeeze(), output_depths.squeeze()
            loss = self.criterion(output_betas, betas) * self.alpha + tv_norm(output_betas, self.lambda0) + \
                    self.criterion(output_depths, depths) * self.lambda1
        elif self.model_name == '2HDEDNet':
            output_depths, output_aif_images = self(images)
            output_depths = output_depths.squeeze()
            L_depth = self.L1Loss(output_depths, depths) + tv_norm(output_depths, 0.001)
            L_image = CharbonnierLoss(images, output_aif_images, 1) + ssim(output_aif_images, images) * 4
            loss = L_depth + 0.01 * L_image
        elif self.model_name == 'D3Net':
            output_betas, output_depths = self(images)
            output_betas, output_depths = output_betas.squeeze(), output_depths.squeeze()
            loss = self.criterion(output_depths, depths)
        elif self.model_name == 'KWANet':
            output_betas, output_depths_inv, W = self(images)
            output_betas, output_depths_inv = output_betas.squeeze(), output_depths_inv.squeeze()
            output_depths, depths_inv = 1 / output_depths_inv, 1 / depths
            loss = self.criterion(output_betas, betas) * self.alpha + tv_norm(images, self.lambda0) + \
                   (self.criterion(output_depths_inv, depths_inv) + tv_norm(torch.abs(W), self.lambda1)) * self.mu
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=self.min_learning_rate
        )
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        images, betas, depths = batch
        if self.model_name in ['KDNet', 'D3Net']:
            output_betas, output_depths = self(images)
            output_betas, output_depths = output_betas.squeeze(), output_depths.squeeze()
        elif self.model_name == '2HDEDNet':
            output_depths, output_aif_images = self(images)
            output_betas, output_depths = output_depths.squeeze(), output_depths.squeeze()
        elif self.model_name == 'KWANet':
            output_betas, output_depths_inv, W = self(images)
            output_betas, output_depths_inv = output_betas.squeeze(), output_depths_inv.squeeze()
            output_depths = 1 / output_depths_inv
        mse_blur, tv_blur, mse_depth = self.criterion(output_betas, betas) ** 0.5, tv_norm(images, 1), self.criterion(output_depths, depths) ** 0.5
        AbsRel, delta1, delta2, delta3 = cal_AbsRel(depths, output_depths), cal_delta(depths, output_depths, 1), cal_delta(depths, output_depths, 2), cal_delta(depths, output_depths, 3)
        loss = mse_blur + mse_depth + tv_blur
        self.log('test_loss', loss)
        self.log('test_rmse_blur', mse_blur)
        self.log('test_rmse_depth', mse_depth)
        self.log('test_AbsRel', AbsRel)
        self.log('test_delta_1', delta1)
        self.log('test_delta_2', delta2)
        self.log('test_delta_3', delta3)
        # visualize_sample(images[0], betas[0], output_betas[0], depths[0], output_depths[0], self.logger, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        images, betas, depths = batch
        if self.model_name in ['KDNet', 'D3Net']:
            output_betas, output_depths = self(images)
            output_betas, output_depths = output_betas.squeeze(), output_depths.squeeze()
            loss = self.criterion(output_betas, betas) * self.alpha + tv_norm(output_betas, self.lambda0) + \
                    self.criterion(output_depths, depths) * self.lambda1
        elif self.model_name == '2HDEDNet':
            output_depths, output_aif_images = self(images)
            output_betas, output_depths = output_depths.squeeze(), output_depths.squeeze()
            L_depth = self.L1Loss(output_depths, depths) + tv_norm(output_depths, 0.001)
            L_image = CharbonnierLoss(images, output_aif_images, 1) + ssim(output_aif_images, images) * 4
            loss = L_depth + 0.01 * L_image
        elif self.model_name == 'KWANet':
            output_betas, output_depths_inv, W = self(images)
            output_betas, output_depths_inv = output_betas.squeeze(), output_depths_inv.squeeze()
            output_depths, depths_inv = 1 / output_depths_inv, 1 / depths
            loss = self.criterion(output_betas, betas) * self.alpha + tv_norm(images, self.lambda0) + \
                   (self.criterion(output_depths_inv, depths_inv) + tv_norm(torch.abs(W), self.lambda1)) * self.mu
        self.log('val_loss', loss)
        if batch_idx < 2:
            visualize_sample(images[0], betas[0], output_betas[0], depths[0], output_depths[0], self.logger, self.global_step)
        return loss