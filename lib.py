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

    
########### print download progress ############
def _print_download_progress(count, block_size, total_size):
    """
    Function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """

    # percentage completion.
    pct_complete = float(count * block_size) / total_size

    # status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # print the message.
    sys.stdout.write(msg)
    sys.stdout.flush()
