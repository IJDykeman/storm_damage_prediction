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
import random

# %matplotlib inline

print sklearn.metrics.roc_auc_score

data_handler = data_loading.DataHandler()
# load the model and restore the weights from the latest checkpoint
tf.reset_default_graph()
model = model.Model()
model.restore()

pool3 = model.vgg_layers['vgg_16/pool3']





metamat, extra_features, y_true = data_handler.get_batch(64, is_validation=False)

pool3_vals = model.sess.run(pool3,
    feed_dict={model.heightmap_ph:metamat,
    model.extra_features_ph:extra_features,
    model.keep_prob_ph: 1})

print "pool3_vals.shape", pool3_vals.shape

nonzero_activation_fraction = 1.0 * np.sum(np.greater(pool3_vals, 0)) / np.prod(pool3_vals.shape)
print "nonzero fraction", nonzero_activation_fraction
print "expected nonzero elements per receptive location", nonzero_activation_fraction * pool3_vals.shape[-1]

np.random.seed(1234)
num_buckets = 32
original_index_to_bucket = np.random.randint(0, num_buckets, [pool3_vals.shape[-1]])

plt.hist(pool3_vals.flatten())
plt.show()

def normed_hash(receptive_location_value):
    assert len(receptive_location_value.shape) == 1
    result = np.zeros([num_buckets])
    for i in range(len(receptive_location_value)):
        result[original_index_to_bucket[i]] += receptive_location_value[i]
    result /= np.sqrt(np.sum(result **2))
    return result

feature_map_width = pool3_vals.shape[1]
receptive_field_width = 44

images = []
hashed_descriptors = []

for i in range(len(pool3_vals)):
	for x in range(feature_map_width):
		for y in range(feature_map_width):
			image = metamat[i, 22 + x * 224/feature_map_width:22 + (x + 1) * 224/feature_map_width, 22 + y * 224/feature_map_width:22 + (y + 1) * 224/feature_map_width]
# print normed_hash(pool3_vals[0,1,1]).shape
# print normed_hash(pool3_vals[0,1,1])
