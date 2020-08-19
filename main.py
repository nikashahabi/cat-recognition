import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

train_file = h5py.File('train_catvnoncat.h5', 'r')
test_file = h5py.File('test_catvnoncat.h5', 'r')
training_dataset_x_initial = np.array(train_file['train_set_x'])
training_dataset_y = np.array(train_file['train_set_y'])
test_dataset_x_initial = np.array(test_file['test_set_x'])
test_dataset_y = np.array(test_file['test_set_y'])
training_dataset_y = training_dataset_y.reshape(1,209)
test_dataset_y = test_dataset_y.reshape(1,50)
index = 25
print(training_dataset_y[:, index])
plt.imshow(training_dataset_x_initial[index])
plt.show()
print(str(training_dataset_y[:, index]))
