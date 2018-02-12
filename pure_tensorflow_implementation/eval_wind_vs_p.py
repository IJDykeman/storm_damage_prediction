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

data_handler = data_loading.DataHandler()
# load the model and restore the weights from the latest checkpoint
model = model.Model()
model.restore()

test_indices = data_handler.test_indices[:]
import random
random.seed(123)
random.shuffle(test_indices)
batch_size = 64

np.random.seed(1)
for prediction_index in np.random.choice(len(test_indices) - 1, 5):
    predictions = []
    batch_indices = [test_indices[prediction_index]]
    metamat, wind_speed, wind_dir, class_y  = data_handler.get_data_batch_from_indices(batch_indices)


    metamat = [metamat[0]] * batch_size
    wind_speed = np.array([wind_speed[0]] * batch_size)
    wind_dir = np.array([wind_dir[0]] * batch_size)
    class_y = [class_y[0]] * batch_size

    for i in range(batch_size):
        wind_speed[i] = np.ones_like(wind_speed[i]) - .5
        wind_speed[i] *= 14.0 * i / batch_size - 1

    pred_probabilities = model.sess.run(model.pred_probabilities,
        feed_dict={model.heightmap_ph:metamat,
        # model.extra_features_ph:extra_features,
        model.wind_speed_placeholder: wind_speed,
        model.wind_direction_placeholder: wind_dir,        
        model.labels_ph:class_y,
        model.keep_prob_ph: 1})
    predictions.extend(pred_probabilities[:, 1])

    print predictions

    plt.plot(1.0 * np.arange(len(predictions)) / len(predictions), predictions)
    plt.xlabel("scaled wind force")
    plt.ylabel("P(damage)")
plt.show()