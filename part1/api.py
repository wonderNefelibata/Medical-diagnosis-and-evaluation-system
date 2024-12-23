import torch
from part1.autoencoder import AutoEncoder
from part1.cnn import CNN
from torchvision import transforms
from PIL import Image


# 处理图像的预处理函数
def transform_image(img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图（1个通道）
        transforms.Resize((256, 256)),  # 调整图像尺寸
        transforms.ToTensor(),  # 转换为Tensor
        # 如果模型在训练时进行了标准化，可以在这里添加标准化步骤
        # transforms.Normalize(mean=[0.5], std=[0.5]),  # 如果是灰度图，可以使用类似这样的标准化
    ])

    img = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式，防止异常图像
    img = transform(img)  # 进行预处理转换
    return img


def diagnosis(img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 创建autoencoder对象
    autoencoder = AutoEncoder().to(device)
    autoencoder.load_state_dict(torch.load("model/autoencoder.pth", map_location=device,
                                           weights_only=True))
    # 创建cnn对象
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load("model/cnn.pth", map_location=device,weights_only=True))

    # 将图片转换为 batch_size 为 1 的张量
    img = img.unsqueeze(0).to(device)  # 增加 batch 维度，形状变为 [1, 1, 256, 256]

    # 使用autoencoder去噪
    input_img = autoencoder(img)
    # 使用cnn判断
    output_result = cnn(input_img)
    # 返回判断结果
    pred = output_result.argmax(dim=1, keepdims=True)
    return pred.item()

# temp_path = "../dataset/covid19/noisy_test/Normal/0102.jpeg"
# # 调用transform_image进行预处理
# img_tensor = transform_image(temp_path).unsqueeze(0)  # 添加批次维度
# result = diagnosis(img_tensor)
# print(result)