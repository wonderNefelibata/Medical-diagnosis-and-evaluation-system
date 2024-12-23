import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def calculate_mean_std(data_path):
    """
    计算数据集的均值和标准差
    :param data_path:
    :return:
    :param data_path:
    :return: mean, std
    """
    dataset = datasets.ImageFolder(data_path, transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]))

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean.item(), std.item()

def get_dataset(data_path):
    """
    读取数据，对train,test通用
    使用ImageFolder将一个文件夹下的图片读取成数据集。
    :param data_path:
    :return: datasets.ImageFolder()
    """
    mean, std = calculate_mean_std(data_path)
    dataset = datasets.ImageFolder(data_path, transforms.Compose([transforms.Grayscale(),
                                                                  transforms.Resize((256, 256)),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize(mean=[mean],
                                                                                       std=[std])]))
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
