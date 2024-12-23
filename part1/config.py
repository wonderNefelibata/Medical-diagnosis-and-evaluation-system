# 定义超参
hyper_parameters = {
    "TRAIN_BATCH_SIZE": 32,
    "TEST_BATCH_SIZE": 66,  # 一次性将所有的测试数据传入
    "EPOCHS": 30,
    "LR": 0.001
}

# 图像路径
data_path = {
    "train_dataset": "../dataset/covid19/train",
    "test_dataset": "../dataset/covid19/noisy_test"
}
