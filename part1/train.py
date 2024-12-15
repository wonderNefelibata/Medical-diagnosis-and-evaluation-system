import torch
import torch.nn as nn
from tqdm import tqdm

from part1.autoencoder import AutoEncoder
from part1.cnn import CNN
from part1.data_tools import get_dataset, get_dataloader, add_noise
from config import hyper_parameters, data_path
from visualization import visualization_for_cnn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Running on " + device)

    train_dataset = get_dataset(data_path["train_dataset"])
    test_dataset = get_dataset(data_path["test_dataset"])

    # 获取dataloader
    train_dataloader = get_dataloader(train_dataset, batch_size=hyper_parameters["TRAIN_BATCH_SIZE"], shuffle=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=hyper_parameters["TEST_BATCH_SIZE"], shuffle=True)

    # 创建自编码器对象
    autoencoder = AutoEncoder().to(device)

    # 从配置文件中读取超参
    EPOCHS = hyper_parameters["EPOCHS"]
    LR = hyper_parameters["LR"]

    # 指定损失函数与优化器算法
    criterion = nn.MSELoss()  # 均方差损失
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

    # 开始训练autoencoder
    autoencoder.train(mode=True)
    for epoch in range(EPOCHS):
        total_loss = 0
        for img, label in tqdm(train_dataloader):
            img = img.to(device)
            noisy_img = add_noise(img)  # 给图片加上高斯噪声
            optimizer.zero_grad()
            reconstructed = autoencoder(noisy_img)  # 使用自编码器进行编码再解码
            loss = criterion(reconstructed, img)  # 比较经过自编码器解码的图像与目标干净图像的差距
            total_loss += loss.item()
            loss.backward()  # TODO：损失函数反向传播，计算梯度，梯度存在哪了？
            optimizer.step()  # TODO：优化器从哪获取到梯度进行更新？
        avg_loss = total_loss / len(train_dataloader)
        print(f'[Autoencoder]  Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}')

    # 保存模型参数
    # torch.save(autoencoder.state_dict(), "../model/autoencoder.pth")

    # 创建CNN对象
    cnn = CNN().to(device)
    # 对于CNN，将损失函数换成交叉熵
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=hyper_parameters["LR"])

    train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch, test_acc_per_epoch = [], [], [], []

    # 开始训练cnn
    # cnn.train(mode=True)
    for epoch in range(EPOCHS):
        cnn.train(mode=True)
        total_loss = 0
        correct = 0
        for img, label in tqdm(train_dataloader):
            img, label = img.to(device), label.to(device)
            noisy_img = add_noise(img)  # 给图片加上高斯噪声
            with torch.no_grad():
                input_img = autoencoder(noisy_img)
            output_result = cnn(input_img)
            optimizer.zero_grad()

            loss = criterion(output_result, label)  # 比较经过自编码器解码的图像与目标干净图像的差距
            total_loss += loss.item()
            pred = output_result.argmax(dim=1, keepdims=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct / len(train_dataset)
        train_loss_per_epoch.append(avg_loss)
        train_acc_per_epoch.append(accuracy)
        # print(f'[CNN]  Epoch [{epoch + 1}/{EPOCHS}] Train, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        cnn.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for img, label in tqdm(test_dataloader):
                img, label = img.to(device), label.to(device)
                input_img = autoencoder(img)
                output_result = cnn(input_img)

                loss = criterion(output_result, label)  # 比较经过自编码器解码的图像与目标干净图像的差距
                total_loss += loss.item()
                pred = output_result.argmax(dim=1, keepdims=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

            avg_loss = total_loss / len(test_dataloader)
            accuracy = correct / len(test_dataset)
            test_loss_per_epoch.append(avg_loss)
            test_acc_per_epoch.append(accuracy)
            print(f'[CNN]  Epoch [{epoch + 1}/{EPOCHS}] Test, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    # 模型保存
    torch.save(cnn.state_dict(), "../model/cnn.pth")
    visualization_for_cnn(list(range(1, EPOCHS + 1)), train_loss_per_epoch, test_loss_per_epoch, train_acc_per_epoch,
                          test_acc_per_epoch)


if __name__ == '__main__':
    main()
