# this is to make the environment compatible with the Atom Hydrogen
# interactive editor package.  You can ignore this code if that doesn't mean
# anything to you.

PYSOLR_PATH = '/home/isaac/Desktop/storm_damage_prediction/pure_tensorflow_implementation'

import model
reload(model)
import data_loading
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


data_handler = data_loading.DataHandler()

wind_direction_data = data_handler.load_data("/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad100_hist8x8.mat.hd",
              normalize_columns=False, only_columns_containing = "dir")

tf.reset_default_graph()
model = model.Model()
model.restore()




ax = plt.axes()
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

index = 100045
for col in wind_direction_data.columns:
    angle = (wind_direction_data[col][index] - 90) / 360 * 2 * np.pi


    batch_indices = [index]
    metamat, extra_features, y_true = data_handler.get_data_batch_from_indices(batch_indices)


    plt.imshow(metamat[0], cmap = 'viridis')
    probabilities = get_saliency_map(metamat[0], extra_features[0], 16, 5, show = False)
    ax.imshow(probabilities)    

    ax.arrow(0, 0, np.cos(angle), np.sin(angle), head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.show()