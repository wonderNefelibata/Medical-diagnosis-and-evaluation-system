import os
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset,random_split
import torch
from tqdm import tqdm
from data_tools import preprocess_data
from part2 import config
from part2.model import Classifier

from visualization import plot_history

# 设置环境变量以解决OpenMP冲突
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载数据
train_file = config.TRAIN_FILE
test_file = config.TEST_FILE
X_train, y_train, X_test, y_test, tokenizer = preprocess_data(train_file, test_file)

X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train.argmax(axis=1), dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.argmax(axis=1), dtype=torch.long)

# 创建DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 创建模型
model = Classifier()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型时的其他函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=1, validation_split=0.2):
    model.train()

    # 从 train_loader 中提取所有数据
    train_data = list(train_loader.dataset)
    train_size = len(train_data)

    # 计算训练集和验证集的大小
    val_size = int(train_size * validation_split)
    train_size = train_size - val_size

    # 使用 random_split 拆分数据集
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=train_loader.batch_size, shuffle=False)

    # 用于记录训练和验证的历史
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # 训练循环
    for epoch in range(num_epochs):
        # 训练部分
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算训练损失和准确率
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        history["loss"].append(train_loss)
        history["accuracy"].append(train_accuracy)

        # 验证部分
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 计算验证准确率
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算验证损失和准确率
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # 打印当前epoch的结果
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 返回历史记录
    return history


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy}")
    return accuracy


# 训练模型
history = train_model(model, train_loader, criterion, optimizer, num_epochs=10,validation_split=0.25)

# 保存模型
torch.save(model.state_dict(), "../model/classifier.pth")

# 评估模型
evaluate_model(model, test_loader)

plot_history(history)
