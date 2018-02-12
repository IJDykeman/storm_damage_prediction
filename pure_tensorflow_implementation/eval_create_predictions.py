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
print sklearn.metrics.roc_auc_score

data_handler = data_loading.DataHandler()

all_indices = range(len(data_handler.meta))

# load the model and restore the weights from the latest checkpoint
model = model.Model()
model.restore()

import time

batch_size = 54
predictions = []
ground_truth = []
for i in range(len(all_indices) / batch_size):
    print i
    batch_indices = all_indices[i * batch_size:(i+1) * batch_size]
    if len(batch_indices) == 0:
        break
    t0 = time.time()
    metamat, wind_speed, wind_dir, class_y  = data_handler.get_data_batch_from_indices(batch_indices)
    # print "_" * 50
    # print batch_indices
    # print metamat, wind_speed, wind_dir, class_y
    t1 = time.time()
        
    ground_truth.extend(class_y[:, 1])
    pred_probabilities = model.sess.run(model.pred_probabilities,
        feed_dict={model.heightmap_ph:metamat,
        # model.extra_features_ph:extra_features,
        model.wind_speed_placeholder: wind_speed,
        model.wind_direction_placeholder: wind_dir,        
        model.labels_ph:class_y,
        model.keep_prob_ph: 1})
    t2 = time.time()
    print t1 - t0, t2 - t0
    predictions.extend(zip(data_handler.meta['hcad'][batch_indices], pred_probabilities[:, 1]))


np.save("hcad_numers_predictions", np.array(predictions))

# print "auc:", sklearn.metrics.roc_auc_score(ground_truth, np.array(predictions), average='macro', sample_weight=None)
