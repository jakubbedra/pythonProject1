import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import random
from urllib.request import urlopen


def download_dataset(file_id):
    filename = f"dataset{file_id}.h5"
    base_url = f"https://github.com/pa-k/AGU/blob/main/assignment1/{filename}?raw=true"
    url = urlopen(base_url)
    binary_data = url.read()
    with open(filename, "wb") as f:
        f.write(binary_data)


def import_dataset(file_id):
    filename = f"dataset{file_id}.h5"
    if not os.path.exists(filename):
        download_dataset(file_id)
    fp = h5py.File(filename, "r")
    x_train = np.array(fp["x_train"][:])
    y_train = np.array(fp["y_train"][:])
    x_test = np.array(fp["x_test"][:])
    y_test = np.array(fp["y_test"][:])
    fp.close()
    return x_train, y_train, x_test, y_test


student_id = 182437  # Your id
file_id = student_id % 17

x_train, y_train, x_test, y_test = import_dataset(file_id)

assert x_train.shape == (600, 32, 32, 3)
assert x_test.shape == (200, 32, 32, 3)
assert y_train.shape == (600, 1)
assert y_test.shape == (200, 1)

index = random.randint(0, x_train.shape[0] - 1)
plt.axis('off')
plt.imshow(x_train[index])
print("Label: " + str(y_train[index]))

x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255

print("Dataset dimensions:")
print("Number of training examples: m_train = " + str(x_train.shape[1]))
print("Number of testing examples: m_test = " + str(x_test.shape[1]))
print("train_x shape: " + str(x_train.shape))
print("train_y shape: " + str(y_train.shape))
print("test_x shape: " + str(x_test.shape))
print("test_y shape: " + str(y_test.shape))


class NeuralNet(object):

    def __init__(self, input_size, hidden_layer_size, output_layer_size):
        """
        input size - number of features of the input
        hidden_layer_size - number of neurons in the hidden layer
        output_layer_size - number of neurons in the output layer (1 in case of this assignment)
        """
        # YOUR CODE HERE
        # Fix definition of weight matrices and biases, use np.random.uniform function
        W1 = np.random.uniform(-0.1, 0.1, (input_size, hidden_layer_size))
        b1 = np.random.uniform(-0.1, 0.1, (1, hidden_layer_size))
        W2 = np.random.uniform(-0.1, 0.1, (hidden_layer_size, output_layer_size))
        b2 = np.random.uniform(-0.1, 0.1, (1, output_layer_size))
        # END OF YOUR CODE
        self.params = {'W1': W1,
                       'b1': b1,
                       'W2': W2,
                       'b2': b2}

        self.grads = {}

    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # YOUR CODE HERE
        # write code to do prediction: for input set of images (dimension [batch,features]) it should set predictions vector [batch] indicatating image class

        # Forward pass

        # hidden layer
        z1 = np.dot(X, W1) + b1  # wagi + bias
        a1 = np.maximum(0, z1)  # ReLU

        # output layer
        z2 = np.dot(a1, W2) + b2
        predictions = 1 / (1 + np.exp(-z2))

        # END OF YOUR CODE
        return np.round(predictions)

    def propagate(self, X, y):
        y = y.reshape(-1, 1)  # fix y vector to have shape [batch,1]
        nbatch = y.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # forward pass
        z1 = np.dot(X, W1) + b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, W2) + b2
        y_hat = 1 / (1 + np.exp(-z2))  # sigmoid
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).sum(axis=0) / nbatch  # cross-entropy loss

        # backward pass
        dz2 = y_hat - y
        dW2 = (1 / nbatch) * np.dot(a1.T, dz2)
        db2 = (1 / nbatch) * np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (z1 > 0)  # relu derivative for first layer
        dW1 = (1 / nbatch) * np.dot(X.T, dz1)
        db1 = (1 / nbatch) * np.sum(dz1, axis=0, keepdims=True)

        self.grads = {'dW1': dW1,
                      'db1': db1,
                      'dW2': dW2,
                      'db2': db2}

        return loss

    def update(self, alpha):
        # YOUR CODE HERE
        # calculate new values of network parames (self.params) based on learning rate alpha and gradients (self.grads)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        dW1 = self.grads['dW1']
        db1 = self.grads['db1']
        dW2 = self.grads['dW2']
        db2 = self.grads['db2']

        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        self.params = {'W1': W1,
                       'b1': b1,
                       'W2': W2,
                       'b2': b2}
        pass


# END OF YOUR CODE

alpha = 0.01
net = NeuralNet(x_train.shape[1], 20, 1)
for i in range(10000):
    loss = net.propagate(x_train, y_train)
    net.update(alpha)
    if i % 100 == 0:
        y_pred_train = net.predict(x_train)
        y_pred_test = net.predict(x_test)
        print(loss)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))
