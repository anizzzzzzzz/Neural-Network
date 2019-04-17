import os
import struct
import numpy as np

from src.MultilayerANN import get_path


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


####### Loading Data
DATA_PATH = os.path.join(get_path(), 'data')

X_train, y_train = load_mnist(DATA_PATH, kind='train')
print('Rows : %d, columns : %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist(DATA_PATH, kind='t10k')
print('Rows : %d, columns : %d' % (X_test.shape[0], X_test.shape[1]))


#### Saving the loaded data for efficiency in numpy zip files
np.savez_compressed(os.path.join(get_path(),'mnist_scaled.npz'),
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)
