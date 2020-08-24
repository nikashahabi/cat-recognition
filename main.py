import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import math


def get_datasets():
    train_file = h5py.File('train_catvnoncat.h5', 'r')
    test_file = h5py.File('test_catvnoncat.h5', 'r')
    training_dataset_x_initial = np.array(train_file['train_set_x'])
    training_dataset_y = np.array(train_file['train_set_y'])
    test_dataset_x_initial = np.array(test_file['test_set_x'])
    test_dataset_y = np.array(test_file['test_set_y'])
    m_train = training_dataset_x_initial.shape[0]
    m_test = test_dataset_x_initial.shape[0]
    #pix = training_dataset_x_initial[1]
    training_dataset_y = training_dataset_y.reshape(1, m_train)
    test_dataset_y = test_dataset_y.reshape(1, m_test)
    training_dataset_x = (training_dataset_x_initial.reshape(training_dataset_x_initial.shape[0], -1).T) / 255
    test_dataset_x = (test_dataset_x_initial.reshape(test_dataset_x_initial.shape[0], -1).T) / 255
    #dim = training_dataset_x.shape[0]
    return training_dataset_x, training_dataset_y, test_dataset_x, test_dataset_y


def sigmoid(z):
    sigmoid = 1/(1+np.exp(-1*z))
    return sigmoid


def initialize_parameters_to_zero(dim):
    return np.zeros((dim, 1)), 0


def propagation(w, b, X, Y):
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


def train_parameters(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagation(w, b, X, Y)
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


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False, draw_cost=False):
    w, b = initialize_parameters_to_zero(X_train.shape[0])
    parameters, grads, costs = train_parameters(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    # print(Y_test)
    # print(Y_prediction_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    if draw_cost:
        draw_costs(costs, learning_rate)
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

def show_an_example(index, dataset_x, dataset_y):
    print(dataset_y[:, index])
    pixel = int(math.sqrt(dataset_x.shape[0]/3))
    new_dataset_x = dataset_x.reshape(dataset_x.shape[1], pixel, pixel, 3)
    plt.imshow(new_dataset_x[index])
    plt.show()

def draw_costs(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


training_dataset_x, training_dataset_y, test_dataset_x, test_dataset_y = get_datasets()
# show_an_example(20,test_dataset_x, test_dataset_y)
model(training_dataset_x, training_dataset_y, test_dataset_x, test_dataset_y, 2000, 0.005, True, True)



