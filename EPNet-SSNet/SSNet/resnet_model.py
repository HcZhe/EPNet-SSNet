import os
from my_dataset import MyDataSet
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch
from utils import train_one_epoch, evaluate, cal_m
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def train(train_images_path, train_images_label, val_images_path=None, val_images_label=None, device='cuda:0',
          classes=5, epochs=20, val=True, batch_size=64, init=True):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if val_images_label is None:
        val_images_label = []
    if val_images_path is None:
        val_images_path = []
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     # transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   # transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    if val:
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])

    train_num = len(train_dataset)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
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

        val_num = len(val_dataset)
        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))
    model = resnet50()
    if init:
        model_weight_path = "./init/resnet50-pre.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 每step_size个epochs将学习率乘以0.1
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        val_loss, val_acc = 0, 0
        if val:
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)
            
        scheduler.step()  # 更新学习率
        
        f = open('./res_dir/train_res/resnet50_res.txt', 'a')
        f.write('epoch: ' + str(epoch + 1) + '\n')
        f.write("train_loss: " + str(round(train_loss, 4)) + '\n')
        f.write("train_acc: " + str(round(train_acc, 4)) + '\n')
        f.write("val_loss: " + str(round(val_loss, 4)) + '\n')
        f.write("val_acc: " + str(round(val_acc, 4)) + '\n\n')
        f.close()
        if best_acc < train_acc:
            best_acc = train_acc
            best_epoch = epoch
            torch.save(model.state_dict(), './res_dir/weights/best_resnet50_model.pth')
    f1 = open('./res_dir/best_res/best.txt', 'a')
    f1.write('res ' + 'epoch ' + str(best_epoch) + '  ' + str(best_acc) + '\n')


def predict(images_path, images_label, num_class=5, batch_size=8):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(224),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    dataset = MyDataSet(images_path=images_path,
                        images_class=images_label,
                        transform=data_transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=nw,
                                         collate_fn=dataset.collate_fn)
    model_weight_path = './res_dir/weights/best_resnet50_model.pth'
    model = resnet50(num_classes=num_class).to(device)
    # print(model)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    score = cal_m(model=model, data_loader=loader, device=device, alo='resnet')
    
        # score 是一个包含两个元素的元组，第一个是一个包含四个元素的元组（评估指标），第二个是一个混淆矩阵
    metrics, confusion_mat = score
    # 分别保存评估指标和混淆矩阵到CSV文件
    metrics_df = pd.DataFrame([metrics], columns=['Accuracy', 'Recall', 'Precision', 'F1'])
    metrics_df.to_csv('./res_dir/label/swin_metrics.csv', header=True, index=False)

    # 将混淆矩阵转换为DataFrame并保存
    confusion_mat_df = pd.DataFrame(confusion_mat, index=['Class {}'.format(i) for i in range(confusion_mat.shape[0])],
                                    columns=['Predicted {}'.format(j) for j in range(confusion_mat.shape[1])])
    confusion_mat_df.to_csv('./res_dir/label/swin_confusion_matrix.csv')    
    pd.DataFrame(np.array(score)).to_csv('./res_dir/label/res' + '.csv', header=None, index=None)
    return score
