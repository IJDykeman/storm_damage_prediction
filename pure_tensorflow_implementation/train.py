PYSOLR_PATH = '/home/isaac/Desktop/storm_damage_prediction/pure_tensorflow_implementation'
# %matplotlib inline

import sys
if not PYSOLR_PATH in sys.path:
    sys.path.append(PYSOLR_PATH)

import model
import data_loading
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pyproj
model = model.Model(tag = 'no_training_feature_extractors')
data_handler = data_loading.DataHandler()
print data_handler.hcad.shape
print data_handler.wind_data.shape


i=0
for _ in range(450):
    print i
    metamat, extra_features, class_y = data_handler.get_batch(64, is_validation=False)

    _,loss_val,_, summary_val = \
        model.sess.run([model.optimizer, model.loss, model.accuracy, model.merged_summaries],
        feed_dict={model.heightmap_ph:metamat,
        model.extra_features_ph:extra_features,
        model.labels_ph:class_y,
        model.keep_prob_ph: .7})
    if i % 5 == 0:
        model.save(i)
        model.train_summary_writer.add_summary(summary_val, i)
    i += 1
