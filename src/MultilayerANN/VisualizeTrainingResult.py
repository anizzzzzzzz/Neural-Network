import os

import matplotlib.pylab as plt

def show_training_cost_graph(epochs, eval_, saving_path):
    plt.plot(range(epochs), eval_['cost'])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(saving_path, 'model_cost.png'))


def show_training_validation_graph(epochs, eval_, saving_path):
    plt.plot(range(epochs), eval_['train_acc'], label='training')
    plt.plot(range(epochs), eval_['valid_acc'], label='validation', linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(os.path.join(saving_path, 'model_training_validation.png'))