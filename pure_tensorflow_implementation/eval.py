PYSOLR_PATH = '/home/isaac/Desktop/storm_damage_prediction/pure_tensorflow_implementation'
# %matplotlib inline

import sys
if not PYSOLR_PATH in sys.path:
    sys.path.append(PYSOLR_PATH)

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
tf.reset_default_graph()
model = model.Model()


model.restore()
# batch_indices = test_indices[i * batch_size:(i+1) * batch_size]
#
# metamat, extra_features, y_true = data_handler.get_data_batch_from_indices(batch_indices)
# ground_truth.extend(y_true[:, 1])
# plt.hist( np.array(model.sess.run(model.heightmap_image_batch,
#     feed_dict={model.heightmap_ph:metamat,
#     model.extra_features_ph:extra_features,
#     model.labels_ph:y_true,
#     model.keep_prob_ph: 1})).flatten())
# plt.show()


test_indices = data_handler.test_indices[:]
import random
random.shuffle(test_indices)
batch_size = 32
predictions = []
ground_truth = []
for i in range(10):
    print i
    batch_indices = test_indices[i * batch_size:(i+1) * batch_size]

    metamat, extra_features, y_true = data_handler.get_data_batch_from_indices(batch_indices)
    ground_truth.extend(y_true[:, 1])
    pred_probabilities = model.sess.run(model.pred_probabilities,
        feed_dict={model.heightmap_ph:metamat,
        model.extra_features_ph:extra_features,
        model.labels_ph:y_true,
        model.keep_prob_ph: 1})
    predictions.extend(pred_probabilities[:, 1])

print np.array(ground_truth)
print np.array(predictions)
print "auc:", sklearn.metrics.roc_auc_score(ground_truth, np.array(predictions), average='macro', sample_weight=None)

# plt.hist(ground_truth)
plt.hist(predictions)
plt.show()
