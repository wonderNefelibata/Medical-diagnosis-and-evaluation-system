import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_dataset(data_path):
    """
    读取数据，对train,test通用
    使用ImageFolder将一个文件夹下的图片读取成数据集。
    :param data_path:
    :return: datasets.ImageFolder()
    """
    dataset = datasets.ImageFolder(data_path, transforms.Compose([transforms.Grayscale(),
                                                                  transforms.Resize((256, 256)),
                                                                  transforms.ToTensor()]))
    return dataset


def get_dataloader(dataset, batch_size, shuffle=True):
    """
    获取dataloader,对train,test通用
    :param dataset:
    :param batch_size:
    :param shuffle:
    :return: DataLoader()
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def add_noise(img, noise_factor=0.5):
    """
    给干净的训练集图片加上高斯噪声
    :param img:
    :param noise_factor:
    :return: noisy_img
    """
    noisy_img = img + noise_factor * torch.randn_like(img)  # img是一个tensor
    noisy_img = torch.clamp(noisy_img, 0., 1.)  # 把小于0的值取0，把大于1的值取1
    return noisy_img