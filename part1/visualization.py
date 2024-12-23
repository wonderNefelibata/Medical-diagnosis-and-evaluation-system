import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from torchvision.transforms import transforms

from part1.autoencoder import AutoEncoder
from part1.data_tools import get_dataloader, get_dataset
from config import data_path,hyper_parameters


def visualization_for_autoencode(autoencoder,test_dataloader,device):
    for data,_ in test_dataloader:
        data = data.to(device)
        output = autoencoder(data)
        tp = transforms.ToPILImage()
        noisy_img = tp(data[0].cpu())
        reconstructed = tp(output[0].cpu())
        plt.imshow(noisy_img,cmap="gray")
        plt.axis("off")
        plt.show()
        plt.imshow(reconstructed, cmap="gray")
        plt.axis("off")
        plt.show()

def visualization_for_cnn(epoch:list,train_loss_per_epoch:list,test_loss_per_epoch:list,
                          train_acc_per_epoch:list,test_acc_per_epoch:list):
    # 创建一个图形和一个轴
    fig, ax1 = plt.subplots()

    # 绘制accuracy折线图
    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy', color=color)
    ax1.plot(epoch, train_acc_per_epoch, label='train_acc', marker='o', color=color)
    ax1.plot(epoch, test_acc_per_epoch, label='test_acc', marker='o', color='tab:green')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # 创建第二个轴对象，共享同一个x轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)
    ax2.plot(epoch, train_loss_per_epoch, label='train_loss', marker='o', color=color)
    ax2.plot(epoch, test_loss_per_epoch, label='test_loss', marker='o', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # 设置x轴的主刻度间隔
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))

    # 绘制网格线
    ax1.grid(axis='y', linestyle='--', color='gray', linewidth=0.5)

    # 显示图表
    plt.show()


def main():
    # 加载Autoencoder模型
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load("../model/autoencoder.pth",weights_only=True))
    test_dataset = get_dataset(data_path["test_dataset"])
    test_dataloader = get_dataloader(test_dataset,hyper_parameters["TEST_BATCH_SIZE"])
    visualization_for_autoencode(autoencoder,test_dataloader,device="cpu")
    # epochs = list(range(1, 6))
    # train_loss_per_epoch = [0.5, 0.4, 0.3, 0.2, 0.1]  # 示例数据
    # test_loss_per_epoch = [0.6, 0.5, 0.4, 0.3, 0.2]  # 示例数据
    # train_acc_per_epoch = [40, 50, 60, 70, 80]  # 示例数据
    # test_acc_per_epoch = [30, 40, 50, 60, 70]  # 示例数据
    #
    # # 调用函数
    # visualization_for_cnn(epochs, train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch)


if __name__ == '__main__':
    main()