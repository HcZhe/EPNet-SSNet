from typing import Optional
import torch.nn as nn
from torch import Tensor
import os
import math
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate, cal_m
import numpy as np
import pandas as pd


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def _mcfg(**kwargs):
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32)
    cfg.update(**kwargs)
    return cfg


model_cfgs = {
    "regnetx_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13),
    "regnetx_400mf": _mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22),
    "regnetx_600mf": _mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16),
    "regnetx_800mf": _mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16),
    "regnetx_1.6gf": _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    "regnetx_3.2gf": _mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25),
    "regnetx_4.0gf": _mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23),
    "regnetx_6.4gf": _mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17),
    "regnetx_8.0gf": _mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23),
    "regnetx_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19),
    "regnetx_16gf": _mcfg(w0=216, wa=55.59, wm=2.1, group_w=128, depth=22),
    "regnetx_32gf": _mcfg(w0=320, wa=69.86, wm=2.0, group_w=168, depth=23),
    "regnety_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    "regnety_400mf": _mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25),
    "regnety_600mf": _mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    "regnety_800mf": _mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),
    "regnety_1.6gf": _mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    "regnety_3.2gf": _mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    "regnety_4.0gf": _mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    "regnety_6.4gf": _mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    "regnety_8.0gf": _mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),
    "regnety_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    "regnety_16gf": _mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    "regnety_32gf": _mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25)
}


def generate_width_depth(wa, w0, wm, depth, q=8):
    assert wa > 0 and w0 > 0 and wm > 1 and w0 % q == 0
    widths_cont = np.arange(depth) * wa + w0
    width_exps = np.round(np.log(widths_cont / w0) / np.log(wm))
    widths_j = w0 * np.power(wm, width_exps)
    widths_j = np.round(np.divide(widths_j, q)) * q
    num_stages, max_stage = len(np.unique(widths_j)), width_exps.max() + 1
    assert num_stages == int(max_stage)
    assert num_stages == 4
    widths = widths_j.astype(int).tolist()
    return widths, num_stages


def adjust_width_groups_comp(widths: list, groups: list):
    groups = [min(g, w_bot) for g, w_bot in zip(groups, widths)]
    widths = [int(round(w / g) * g) for w, g in zip(widths, groups)]
    return widths, groups


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 kernel_s: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.ReLU(inplace=True)):
        super(ConvBNAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_c,
                              out_channels=out_c,
                              kernel_size=kernel_s,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_c)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RegHead(nn.Module):
    def __init__(self,
                 in_unit: int = 368,
                 out_unit: int = 1000,
                 output_size: tuple = (1, 1),
                 drop_ratio: float = 0.25):
        super(RegHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        self.fc = nn.Linear(in_features=in_unit, out_features=out_unit)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, se_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 stride: int = 1,
                 group_width: int = 1,
                 se_ratio: float = 0.,
                 drop_ratio: float = 0.):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1)
        self.conv2 = ConvBNAct(in_c=out_c,
                               out_c=out_c,
                               kernel_s=3,
                               stride=stride,
                               padding=1,
                               groups=out_c // group_width)

        if se_ratio > 0:
            self.se = SqueezeExcitation(in_c, out_c, se_ratio)
        else:
            self.se = nn.Identity()

        self.conv3 = ConvBNAct(in_c=out_c, out_c=out_c, kernel_s=1, act=None)
        self.ac3 = nn.ReLU(inplace=True)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        if (in_c != out_c) or (stride != 1):
            self.downsample = ConvBNAct(in_c=in_c, out_c=out_c, kernel_s=1, stride=stride, act=None)
        else:
            self.downsample = nn.Identity()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        x = self.dropout(x)
        shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.ac3(x)
        return x


class RegStage(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 depth: int,
                 group_width: int,
                 se_ratio: float):
        super(RegStage, self).__init__()
        for i in range(depth):
            block_stride = 2 if i == 0 else 1
            block_in_c = in_c if i == 0 else out_c

            name = "b{}".format(i + 1)
            self.add_module(name,
                            Bottleneck(in_c=block_in_c,
                                       out_c=out_c,
                                       stride=block_stride,
                                       group_width=group_width,
                                       se_ratio=se_ratio))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.children():
            x = block(x)
        return x


class RegNet(nn.Module):
    def __init__(self,
                 cfg: dict,
                 in_c: int = 3,
                 num_classes: int = 1000,
                 zero_init_last_bn: bool = True):
        super(RegNet, self).__init__()
        stem_c = cfg["stem_width"]
        self.stem = ConvBNAct(in_c, out_c=stem_c, kernel_s=3, stride=2, padding=1)
        input_channels = stem_c
        stage_info = self._build_stage_info(cfg)
        for i, stage_args in enumerate(stage_info):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(in_c=input_channels, **stage_args))
            input_channels = stage_args["out_c"]
        self.head = RegHead(in_unit=input_channels, out_unit=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def _build_stage_info(cfg: dict):
        wa, w0, wm, d = cfg["wa"], cfg["w0"], cfg["wm"], cfg["depth"]
        widths, num_stages = generate_width_depth(wa, w0, wm, d)

        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_groups = [cfg['group_w'] for _ in range(num_stages)]
        stage_widths, stage_groups = adjust_width_groups_comp(stage_widths, stage_groups)

        info = []
        for i in range(num_stages):
            info.append(dict(out_c=stage_widths[i],
                             depth=stage_depths[i],
                             group_width=stage_groups[i],
                             se_ratio=cfg["se_ratio"]))

        return info


def create_regnet(model_name="RegNetX_200MF", num_classes=1000):
    model_name = model_name.lower().replace("-", "_")
    if model_name not in model_cfgs.keys():
        print("support model name: \n{}".format("\n".join(model_cfgs.keys())))
        raise KeyError("not support model name: {}".format(model_name))

    model = RegNet(cfg=model_cfgs[model_name], num_classes=num_classes)
    return model


def train(train_images_path, train_images_label, val_images_path=None, val_images_label=None, classes=2, epochs=20,
          batch_size=32,
          lr=0.001, lrf=0.01, weight='./init/regnety_400mf.pth', freeze=False, device='cuda:0', val=True, data=1, init=True):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if val_images_label is None:
        val_images_label = []
    if val_images_path is None:
        val_images_path = []

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    if val:
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    if val:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)

    model = create_regnet(num_classes=classes, model_name='RegNetY_400MF').to(device)

    if init:
        if os.path.exists(weight):
            weights_dict = torch.load(weight, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weight))

    if freeze:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("train {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()
        val_loss, val_acc = 0, 0
        if val:
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

        f = open('./res_dir/train_res/data'+str(data)+'/regnet_res.txt', 'a')
        f.write('epoch: ' + str(epoch + 1) + '\n')
        f.write("train_loss: " + str(round(train_loss, 4)) + '\n')
        f.write("train_acc: " + str(round(train_acc, 4)) + '\n')
        f.write("val_loss: " + str(round(val_loss, 4)) + '\n')
        f.write("val_acc: " + str(round(val_acc, 4)) + '\n\n')
        f.close()
        if best_acc < train_acc:
            best_acc = train_acc
            best_epoch = epoch
            torch.save(model.state_dict(), './res_dir/weights/data'+str(data)+'/best_regnet_model.pth')
    f1 = open('./res_dir/best_res/best.txt', 'a')
    f1.write('reg ' + 'epoch ' + str(best_epoch) + '  ' + str(best_acc) + '\n')


def predict(images_path, images_label, num_class=2, batch_size=8, data=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = MyDataSet(images_path=images_path,
                        images_class=images_label,
                        transform=data_transform)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=nw,
                                         collate_fn=dataset.collate_fn)
    model = create_regnet(model_name="RegNetY_400MF", num_classes=num_class).to(device)
    model_weight_path = './res_dir/weights/data'+str(data)+'/best_regnet_model.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score = cal_m(model=model, data_loader=loader, device=device, alo='regnet', num=num_class)
    pd.DataFrame(np.array(score)).to_csv('./res_dir/label/data' + str(data) + '/reg' + '.csv', header=None, index=None)
    return score
