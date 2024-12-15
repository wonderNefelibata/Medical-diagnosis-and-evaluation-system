import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def plot_history(history):
    """
    绘制训练和验证的准确率和损失图
    """
    f, ax = plt.subplots(1, 2, figsize=(16, 7))

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
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
