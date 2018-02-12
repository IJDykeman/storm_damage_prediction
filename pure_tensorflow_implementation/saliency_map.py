# this is to make the environment compatible with the Atom Hydrogen
# interactive editor package.  You can ignore this code if that doesn't mean
# anything to you.

PYSOLR_PATH = '/home/isaac/Desktop/storm_damage_prediction/pure_tensorflow_implementation'
import sys
if not PYSOLR_PATH in sys.path:
    sys.path.append(PYSOLR_PATH)
#########

import model
reload(model)
import data_loading
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import sklearn.manifold
import time
from saliency_map_utils import *

# %matplotlib inline

print sklearn.metrics.roc_auc_score

data_handler = data_loading.DataHandler()
# load the model and restore the weights from the latest checkpoint
tf.reset_default_graph()
model = model.Model()
model.restore()




batch_indices = [234678]
metamat, extra_features, y_true = data_handler.get_data_batch_from_indices(batch_indices)


# %matplotlib inline
plt.imshow(metamat[0], cmap = 'viridis')
plt.show()
plt.imshow(get_saliency_map(metamat[0], extra_features[0], 32, 10, model))
plt.show()
# get_saliency_map(metamat[0], extra_features[0], model, 16, 5)
# get_saliency_map(metamat[0], extra_features[0],model 32, 5)
# get_saliency_map(metamat[0], extra_features[0],model 32, 5)
# get_saliency_map(metamat[0], extra_features[0],model 64, 5)

# get_saliency_map(metamat[0], extra_features[0],model 16, 5)
# get_saliency_map(metamat[0], extra_features[0],model 32, 5)
# get_saliency_map(metamat[0], extra_features[0],model 64, 5)

# get_saliency_map(metamat[0], extra_features[0],model 16, 2)
# get_saliency_map(metamat[0], extra_features[0],model 32, 2)
# get_saliency_map(metamat[0], extra_features[0],model 64, 2)
