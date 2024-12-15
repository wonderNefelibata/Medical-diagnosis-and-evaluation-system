import os
from keras.callbacks import EarlyStopping
from model import create_model
from data_tools import preprocess_data
from visualization import plot_history
import config

# 设置环境变量以解决OpenMP冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载数据
train_file = config.TRAIN_FILE
test_file = config.TEST_FILE
X_train, y_train, X_test, y_test, tokenizer = preprocess_data(train_file, test_file)

# 创建模型
model = create_model()

# 训练模型
history = model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1, validation_split=0.25)

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

# 绘制训练过程图
plot_history(history)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")
