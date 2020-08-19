import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def sigmoid(z):
    sigmoid = 1/(1+np.exp(-1*z))
    return sigmoid


def initialize(dim):
    return np.zeros((dim, 1))


def forward_propagation(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * (np.dot(Y, (np.log(A)).T) + np.dot(1 - Y, (np.log(1 - A)).T))  # compute cost
    dZ = A - Y
    dw = 1 / m * (np.dot(X, dZ.T))
    db = 1 / m * np.sum(dZ)
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def update_parameters(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):

        grads, cost = forward_propagation(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    parameters = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return parameters, grads, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):

        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    assert (Y_prediction.shape == (1, m))
    return Y_prediction

train_file = h5py.File('train_catvnoncat.h5', 'r')
test_file = h5py.File('test_catvnoncat.h5', 'r')
training_dataset_x_initial = np.array(train_file['train_set_x'])
training_dataset_y = np.array(train_file['train_set_y'])
test_dataset_x_initial = np.array(test_file['test_set_x'])
test_dataset_y = np.array(test_file['test_set_y'])
m_train = training_dataset_x_initial.shape[0]
m_test = test_dataset_x_initial.shape[0]
pix = training_dataset_x_initial[1]
training_dataset_y = training_dataset_y.reshape(1, m_train)
test_dataset_y = test_dataset_y.reshape(1, m_test)
index = 25
training_dataset_x = (training_dataset_x_initial.reshape(training_dataset_x_initial.shape[0], -1).T) / 255
test_set_x = (test_dataset_x_initial.reshape(test_dataset_x_initial.shape[0], -1).T) / 255
dim = training_dataset_x.shape[0]
w = initialize(dim)
b = 0
# print(training_dataset_y[:, index])
# plt.imshow(training_dataset_x_initial[index])
# plt.show()


