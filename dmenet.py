import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG19_down(nn.Module):
    def __init__(self, is_test=False):
        super(VGG19_down, self).__init__()
        self.is_test = is_test
        self.VGG_MEAN = [103.939, 116.779, 123.68]

        # Define the layers for VGG19
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        if not self.is_test:
            self.logits = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(512, 64, kernel_size=3, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 512, kernel_size=(3, 3), padding=0),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(512*3*3, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1)
            )

    def forward(self, x):
        x = x * 255.0
        blue, green, red = x[:, 2, :, :], x[:, 1, :, :], x[:, 0, :, :]
        blue -= self.VGG_MEAN[0]
        green -= self.VGG_MEAN[1]
        red -= self.VGG_MEAN[2]
        x = torch.stack([blue, green, red], dim=1)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        if self.is_test:
            return feats[-5:]
        logits = self.logits(x)
        return x, feats[-5:], feats[-2], logits

class UNet_up(nn.Module):
    def __init__(self, is_train=False):
        super(UNet_up, self).__init__()
        self.is_train = is_train
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # Define the layers for UNet
        self.u4 = self._make_layer(512, 256)
        self.u3 = self._make_layer(256, 256)
        self.u3_aux = self._make_aux_layer(256, 128)
        self.u2 = self._make_layer(128, 128)
        self.u2_aux = self._make_aux_layer(128, 64)
        self.u1 = self._make_layer(64, 64)
        self.u1_aux = self._make_aux_layer(64, 32)
        self.u0 = self._make_layer(64, 64)
        self.u0_aux = self._make_aux_layer(64, 32)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.concat = lambda x, y: torch.cat((x, y), dim=1)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            self.lrelu,
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            self.lrelu,
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            self.lrelu
        )

    def _make_aux_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            self.lrelu,
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, feats):
        d0, d1, d2, d3, d4 = feats

        u4 = self.u4(d4)
        u4_aux = self.u4_aux(u4)

        u3 = self.upsample(u4)
        u3 = self.concat(u3, d3)
        u3 = self.u3(u3)
        u3_aux = self.u3_aux(u3)

        u2 = self.upsample(u3)
        u2 = self.concat(u2, d2)
        u2 = self.u2(u2)
        u2_aux = self.u2_aux(u2)

        u1 = self.upsample(u2)
        u1 = self.concat(u1, d1)
        u1 = self.u1(u1)
        u1_aux = self.u1_aux(u1)

        u0 = self.upsample(u1)
        u0 = self.concat(u0, d0)
        u0 = self.u0(u0)
        u0_aux = self.u0_aux(u0)

        refine_lists = [u0]
        for _ in range(7):
            n_res = self._make_layer(64, 64)(u0)
            refine_lists.append(n_res)

        return u4_aux, u3_aux, u2_aux, u1_aux, u0_aux, refine_lists
