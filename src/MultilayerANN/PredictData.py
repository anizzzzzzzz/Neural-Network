import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.MultilayerANN import get_path


NUMPY_ZIPPED_DATA_PATH = os.path.join(get_path(), 'mnist_scaled.npz')

mnist = np.load(NUMPY_ZIPPED_DATA_PATH)

X_test, y_test = mnist['X_test'], mnist['y_test']

filename = 'mnist_multilayer_ann_model'
file_path = os.path.join(get_path(),'model',filename)

#loading model
nn = pickle.load(open(file_path, 'rb'))

y_test_pred = nn.predict(X_test)

# calculating accuracy
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])

print('Training accuracy : %.2f%%' % (acc * 100))

#####################################################

# displaying the data that model failed to predict

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex='all',
                       sharey='all')

ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()