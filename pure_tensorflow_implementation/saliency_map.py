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

# %matplotlib inline

print sklearn.metrics.roc_auc_score

data_handler = data_loading.DataHandler()
# load the model and restore the weights from the latest checkpoint
tf.reset_default_graph()
model = model.Model()
model.restore()

def gray_box(image, row, col, size):
    result = np.copy(image)
    result[row - size/2 : row + size/2, col - size/2 : col + size/2] = np.max(image)
    #     result = scipy.interpolate.interp2d(x, y, z, kind='linear', copy=True, bounds_error=False, fill_value=np.nan)
    return result


batch_indices = [349829]
metamat, extra_features, y_true = data_handler.get_data_batch_from_indices(batch_indices)

def get_saliency_map (image, extra_features_vec, box_size, step):
    pad = box_size/2
    locations = []
    images = []

    for row in range(pad,224 - 2 - pad, step):
        for col in range(pad,224 - 2 - pad, step):
            locations.append((row / step, col / step))
            images.append(gray_box(image, row, col,box_size))
            # plt.imshow(images[-1])
            # plt.show()


    def get_probs(images, extra_features):
        pred_probabilities = model.sess.run(model.pred_probabilities,
            feed_dict={model.heightmap_ph:images,
            model.extra_features_ph:extra_features,
            model.keep_prob_ph: 1})

        return pred_probabilities


    probabilities = []
    batch_size = 10
    for i in range(0, len(locations), batch_size):
        image_batch = np.array(images[i: i+batch_size])
        extra_features_batch = np.array([extra_features_vec] * len(image_batch))
        probs = get_probs(image_batch, extra_features_batch)
        probabilities.extend(probs)


    probabilities = np.array(probabilities)[:,1]
    probabilities.shape
    probabilities = probabilities.reshape([int(np.sqrt(len(probabilities)))]*2)
    plt.imshow(probabilities, cmap='viridis')
    plt.show()
    # plt.imshow(images[0])
    # plt.show()

%matplotlib inline
plt.imshow(metamat[0], cmap = 'viridis')
get_saliency_map(metamat[0], extra_features[0], 16, 4)
get_saliency_map(metamat[0], extra_features[0], 32, 4)
get_saliency_map(metamat[0], extra_features[0], 64, 4)

get_saliency_map(metamat[0], extra_features[0], 16, 5)
get_saliency_map(metamat[0], extra_features[0], 32, 5)
get_saliency_map(metamat[0], extra_features[0], 64, 5)

get_saliency_map(metamat[0], extra_features[0], 16, 2)
get_saliency_map(metamat[0], extra_features[0], 32, 2)
get_saliency_map(metamat[0], extra_features[0], 64, 2)
