import matplotlib.pyplot as plt


########### plot validation history ############
def plot_validation_history(his, fig_path):
    train_loss = his.history['loss']
    val_loss = his.history['val_loss']

    # visualize training history
    plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='Train loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, color='red', label='Val loss')
    plt.legend(loc="upper right")
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    plt.savefig(fig_path, dpi=300)
    plt.show()
