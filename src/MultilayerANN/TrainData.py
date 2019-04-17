import os

import numpy as np
import pickle

from src.MultilayerANN import get_path
from src.MultilayerANN.NeuralNetMLP import NeuralNetMLP
from src.MultilayerANN.VisualizeTrainingResult import show_training_cost_graph, show_training_validation_graph

NUMPY_ZIPPED_DATA_PATH = os.path.join(get_path(), 'mnist_scaled.npz')

mnist = np.load(NUMPY_ZIPPED_DATA_PATH)

# print(mnist.files)
# X_train = mnist['X_train']

X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]

# print('X_train shape : ',X_train.shape)
# print('y_train shape : ', y_train)

# show_plot(X_train, y_train)
# show_plot(X_train, y_train, nrows=5, ncols=5)

nn = NeuralNetMLP(n_hidden=100,
                  l2=0.01,
                  epochs=200,
                  eta=0.0005,
                  minibatch_size=100,
                  shuffle=True,
                  seed=1)

nn.fit(X_train=X_train[:55000],
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])

# saving the model
filename = 'mnist_multilayer_ann_model'
file_path = os.path.join(get_path(),'model',filename)
pickle.dump(nn, open(file_path, 'wb'))

# saving the image of model training cost
show_training_cost_graph(nn.epochs, nn.eval_, os.path.join(get_path(), 'model'))

# saving the image of training and validation accuracy
show_training_validation_graph(nn.epochs, nn.eval_, os.path.join(get_path(), 'model'))
