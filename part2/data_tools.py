# data_tools.py
import pickle

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from part2.config import WORDS, LENGTH


def load_data(file_path):
    """
    加载数据集
    """
    return pd.read_csv(file_path)


def process_labels(rating):
    """
    处理评分标签，返回二分类标签
    """
    return -1 * (rating <= 4) + 1 * (rating >= 7)


def tokenize_reviews(reviews, tokenizer=None,save_path=None):
    """
    对评论进行tokenize操作，返回token序列
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=WORDS)
        tokenizer.fit_on_texts(reviews)
        # 如果指定了保存路径，则保存 tokenizer
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    return tokenizer.texts_to_sequences(reviews), tokenizer


def load_tokenizer(save_path):
    """
    从指定路径加载已保存的tokenizer
    """
    with open(save_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def preprocess_data(train_file, test_file):
    """
    处理训练集和测试集
    """
    # 读取数据
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    # 获取标签
    train_rating = train_data["rating"]
    train_label = process_labels(train_rating)
    test_rating = test_data["rating"]
    test_label = process_labels(test_rating)

    # 处理评论文本
    train_review = train_data["review"]
    test_review = test_data["review"]

    # Tokenize评论
    train_sequence, tokenizer = tokenize_reviews(train_review,save_path="../model/tokenizer.pkl")
    test_sequence, _ = tokenize_reviews(test_review, tokenizer)

    # 填充序列
    X_train = pad_sequences(train_sequence, maxlen=LENGTH)
    X_test = pad_sequences(test_sequence, maxlen=LENGTH)

    # One-hot编码标签
    y_train = to_categorical(train_label, num_classes=3)
    y_test = to_categorical(test_label, num_classes=3)

    return X_train, y_train, X_test, y_test, tokenizer
