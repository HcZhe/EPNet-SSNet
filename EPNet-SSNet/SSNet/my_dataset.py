from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):

        self.images_path = images_path
        self.images_class = images_class
        # 添加一个固定的尺寸调整转换
        self.resize_transform = transforms.Resize((224, 224))
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # print("Attempt to open:", self.images_path[item])  # 添加这行来检查路径
        img = Image.open(self.images_path[item])
        img = img.convert('RGB')
        # 首先调整图像大小
        img = self.resize_transform(img)
        # 应用其他任何传递的转换
        if self.transform is not None:
            img = self.transform(img)
        label = self.images_class[item]
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
