import matplotlib.pyplot as plt

def show_plot(X_train, y_train, nrows=2, ncols=5):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           sharex=True, sharey=True)

    ax = ax.flatten()
    for i in range(10):
    # for i in range(25):
        img = X_train[y_train == i][0].reshape(28, 28)
        # img = X_train[y_train == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()