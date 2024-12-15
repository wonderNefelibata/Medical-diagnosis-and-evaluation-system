from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from part2.config import WORDS, LENGTH, DEPTH
import torch.nn as nn

def create_model():
    """
    创建并编译模型(Keras方法)
    """
    model = Sequential()
    model.add(Embedding(WORDS, DEPTH, input_length=LENGTH))
    model.add(LSTM(DEPTH))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['acc'])

    return model

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(WORDS, DEPTH, padding_idx=0)

        # LSTM 层
        self.lstm = nn.LSTM(DEPTH, DEPTH, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(DEPTH, 3)

    def forward(self, x):
        # 嵌入层
        x = self.embedding(x)

        # LSTM 层
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)

        # 使用最后一个时间步的隐藏状态作为模型的输出
        out = self.fc(hn[-1])  # hn[-1] 是最后一个时间步的隐藏状态

        return out



