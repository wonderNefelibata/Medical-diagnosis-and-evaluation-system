import pandas as pd
from keras.utils import to_categorical, pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 处理训练集
train_data = pd.read_csv("../dataset/review/drugsComTrain_raw.csv")
train_rating = train_data["rating"]
train_label = -1*(train_rating <= 4)+1*(train_rating >= 7)
# print(train_label.head())

# 处理测试集
test_data = pd.read_csv("../dataset/review/drugsComTest_raw.csv")
test_rating = test_data["rating"]
test_label = -1*(test_rating <= 4)+1*(test_rating >= 7)

# 设置一些常量
WORDS = 50000
LENGTH = 170
DEPTH = 64


def plot_history(history):
    f, ax = plt.subplots(1, 2, figsize=(16, 7))  # 改为 subplots()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    ax[0].xaxis.set_major_locator(MultipleLocator(1))
    ax[0].plot(epochs, acc, marker='o', label='Training acc')
    ax[0].plot(epochs, val_acc, marker='o', label='Validation acc')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_title('Training and validation accuracy')
    ax[0].grid(axis="y", linestyle='--')
    ax[0].legend()

    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    ax[1].plot(epochs, loss, marker='o', label='Training loss')
    ax[1].plot(epochs, val_loss, marker='o', label='Validation loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_title('Training and validation loss')
    ax[1].grid(axis="y", linestyle='--')
    ax[1].legend()

    plt.show()

# 对训练数据tokenize
train_review = train_data["review"]
tokenizer = Tokenizer(num_words=WORDS)  # 不设置num_words这个参数的话，就会默认保留所有的单词。可以尝试不同的取值，性能会不一样
# tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_review)

train_sequence = tokenizer.texts_to_sequences(train_review)

X_train = pad_sequences(train_sequence,maxlen=LENGTH)
y_train = to_categorical(train_label,num_classes=3)

# 对于测试集采用同样的处理方式，注意不需要一个新的tokenizer
test_review = test_data["review"]
# tokenizer = Tokenizer(num_words=WORDS)  # 不设置num_words这个参数的话，就会默认保留所有的单词。可以尝试不同的取值，性能会不一样
test_sequence = tokenizer.texts_to_sequences(test_review)

X_test = pad_sequences(test_sequence,maxlen=LENGTH)
y_test = to_categorical(test_label,num_classes=3)

# 搭建模型
model = Sequential()
model.add(Embedding(WORDS,DEPTH,input_length=LENGTH))
model.add(LSTM(DEPTH))
model.add(Dense(3,activation='softmax'))

# rmsprop主要用于推荐系统和NLP，Adam主要用于图像和声音
model.compile(optimizer="rmsprop",loss='categorical_crossentropy',metrics=['acc'])

history = model.fit(X_train,y_train,epochs=1,batch_size=128,verbose=1,validation_split=0.25)

test_loss,test_acc = model.evaluate(X_test,y_test,verbose=1)

plot_history(history)


# <=============================================================================>
# 下面是用来确定句子最长设置为多少token的测试代码
# 对训练数据tokenize
# train_review = train_data["review"]
# tokenizer = Tokenizer(num_words=WORDS)  # 不设置num_words这个参数的话，就会默认保留所有的单词。可以尝试不同的取值，性能会不一样
# tokenizer.fit_on_texts(train_review)
# train_sequence = tokenizer.texts_to_sequences(train_review)
# # result的key表示一个sequence的长度，value表示这个长度的sequence的数量
# result = {}
# for ls in train_sequence:
#     len_of_ls = len(ls)
#     if len_of_ls in result.keys():
#         result[len_of_ls] = result.get(len_of_ls)+1
#     else:
#         result[len_of_ls] = 1
#
# keys = list(result.keys())
# keys.sort()
#
# boundary = 100
# percentage = 0
# total_data = len(train_sequence)
#
# while percentage < 0.99:
#     count = 0
#     boundary += 10
#     for i in keys:
#         if i <= boundary:
#             count += result.get(i)
#     percentage = count / total_data

# count = 0
#
# for value in result.values():
#     if value>count:
#         count = value
#
# for key in result.keys():
#     if result.get(key) == count:
#         LENGTH = key
#         break





