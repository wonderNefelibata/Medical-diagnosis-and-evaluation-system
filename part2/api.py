from keras.utils import pad_sequences
from part2.config import LENGTH
from part2.data_tools import load_tokenizer
from part2.model import Classifier
import torch.nn.functional as F
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def sentiment_classify(review):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载之前保存的 tokenizer
    tokenizer = load_tokenizer("model/tokenizer.pkl")
    # 使用tokenizer进行tokenize
    tokenized_vec = tokenizer.texts_to_sequences([review])
    # 进行填充
    input = pad_sequences(tokenized_vec, maxlen=LENGTH)
    # 将输入转换为torch张量
    input_tensor = torch.tensor(input, dtype=torch.long)  # 默认输入是整数
    # 创建模型
    model = Classifier()
    model.load_state_dict(torch.load("model/classifier.pth", weights_only=True))
    # 输入模型获得结果
    output = model(input_tensor)

    # 应用softmax将logits转化为概率
    probabilities = F.softmax(output, dim=1)  # softmax 作用在类别维度（dim=1）

    # 获取最大概率对应的类别
    predicted_class = torch.argmax(probabilities, dim=1).item()  # 预测类别

    return predicted_class


# review = "Quick reduction of symptoms"
# print(sentiment_classify(review))