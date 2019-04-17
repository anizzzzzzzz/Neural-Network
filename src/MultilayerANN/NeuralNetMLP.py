import numpy as np
import sys

class NeuralNetMLP(object):
    """
    Feedforward neural network / Multi-layer perceptron classifier

    Parameters
    ---------------
    n_hidden : int(default: 30)
        Number of hidden units
    l2 : float (default: 0.)
        Lambda value for L2-regularization
        No regularization if l2=0. (default)
    epochs : int (default:100)
        Number of passes over the training set.
    eta : float(default: 0.001)
        Learning rate
    shuffle : bool (default: True)
        Shuffles training data every epoch
        if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training samples per minibatch
    seed : int (default: None)
        Random seed for initializing weights and shuffling

    Attributes
    -------------------------
    eval_ : dict
        Dictionary collecting the cost, training accuracy,
        and validation accuracy for each epoch during training
    """
    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """
        Encode labels into one-hot representation

        Parameters
        -----------------
        y : array, shape = [n_samples]
            Target values

        Returns
        ------------------
        onehot : array, shape = (n_samples, n_labels)
        """
        # y_train = [5 0 4 ... 5 6 8]
        # creates onehot of zeros with shape (9, 6000)
        # loop will create one hot encoding
        # eg, for 1st :- onehot[5,0] = 1
        #       for 2nd :- onehot[0,1] = 1
        # finally matrix translation is performed and returned.
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        # Given an interval, values outside the interval are clipped to the interval edges.
        # a = np.arange(10)
        # np.clip(a, 3, 6, out=a)
        # array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute logistic function (sigmoid)"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """
        Compute cost function

        Parameters
        ---------------
        :param y_enc: array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        :param output: array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)
        :return:
        ----------
        cost : float
            Regularized cost
        """
        L2_term = (self.l2 * (np.sum(self.w_h ** 2.) + np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        """
        Predict class labels

        :param X: array, shape = [n_samples, n_features]
            Input layer with original features
        :return:
        y_pred : array, shape = [n_samples]
            Predicted class labels.
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        # argmax returns the indices of max element (value)
        # check https://www.geeksforgeeks.org/numpy-argmax-python/
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        Learn weights from training data.

        :param X_train: array, shape = [n_samples, n_features]
            Input layer with original features
        :param y_train: array, shape = [n_samples]
            Target class labels
        :param X_valid: array, shape = [n_samples, n_features]
            Sample features for validation during training
        :param y_valid: array, shape = [n_samples]
            Sample labels for validation during training

        :return:
        ------------------
        self
        """
        # Find the unique elements of an array.
        # np.unique([1, 1, 2, 2, 3, 3]) => array([1, 2, 3])
        # a = np.array([[1, 1], [2, 3]]) = np.unique(a) => array([1, 2, 3])
        n_output = np.unique(y_train).shape[0] # no. of class labels
        n_features = X_train.shape[1]
        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden
        # b_h is bias for each hidden unit
        self.b_h = np.zeros(self.n_hidden)
        # Draw random samples from a normal (Gaussian) distribution.
        # loc => mean, scale => standard deviation
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs)) # for progr. format
        self.eval_ = {'cost' : [], 'train_acc' : [], 'valid_acc' : []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):
            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx  in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx : start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_samples, n_classlabels]
                # ---------------Error term of output layer----------------------
                sigma_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_hidden]
                # derivative of sigmoid function
                # dφ(z)/dz => d/dz (1 / ( 1 + e^(-z) )) => a(1-a)
                sigmoid_derivative_h = a_h * (1. - a_h)

                # first dot product : [n_samples, n_classlabels] dot [n_classlabels, n_samples] (ie;w_out.T)
                # 2nd dot product : [n_samples, n_samples] dot [n_samples, n_hidden]
                # -> [n_samples, n_hidden]
                # ------------------Error term of hidden layer-----------------
                sigma_h = (np.dot(sigma_out, self.w_out.T) * sigmoid_derivative_h)

                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                # Compute the gradient for hidden units
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                # arr1 = [[1,2],[1,2],[1,2]]
                # np.sum(arr1, axis=0) => [3,6]
                grad_b_h = np.sum(sigma_h, axis=0)

                # [n_hidden, n-samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                # Compute the gradient for output units
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and WEIGHT UPDATES
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) / X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f | Train/Valid Acc. : %.2d%%/%.2f%%'
                             % (epoch_strlen, i+1, self.epochs, cost, train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self
