import numpy as np
from skdata.mnist.views import OfficialImageClassification
from matplotlib import pyplot as plt
from tsne import bh_sne
import scipy.io as sio

# # load up data
# data = datOfficialImageClassification(x_dtype="float32")
# print data

# x_data = data.all_images
# y_data = data.all_labels

# load up data
data = sio.loadmat('mnist_train.mat')
print data

x_data = data['train_X']
y_data = data['train_labels']

# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))

# For speed of computation, only run on a subset
#n = 20000
#x_data = x_data[:n]
#y_data = y_data[:n]

# perform t-SNE embedding
vis_data = bh_sne(x_data, theta=.5)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()