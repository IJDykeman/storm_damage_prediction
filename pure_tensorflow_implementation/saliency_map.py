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


# test_indices = data_handler.test_indices[:]
# import random
# random.shuffle(test_indices)
# batch_size = 32
# predictions = []
# ground_truth = []
# for i in range(250):
#     print i
#     batch_indices = test_indices[i * batch_size:(i+1) * batch_size]
#
#     metamat, extra_features, y_true = data_handler.get_data_batch_from_indices(batch_indices)
#     ground_truth.extend(y_true[:, 1])
#     pred_probabilities = model.sess.run(model.pred_probabilities,
#         feed_dict={model.heightmap_ph:metamat,
#         model.extra_features_ph:extra_features,
#         model.labels_ph:y_true,
#         model.keep_prob_ph: 1})
#     predictions.extend(pred_probabilities[:, 1])
#
# print "auc:", sklearn.metrics.roc_auc_score(ground_truth, np.array(predictions), average='macro', sample_weight=None)

#################################
#
# test_indices = data_handler.test_indices[:]
# import random
# random.shuffle(test_indices)
# batch_size = 32
# i=0
# batch_indices = test_indices[i * batch_size:(i+1) * batch_size]
#
# metamat, extra_features, y_true = data_handler.get_data_batch_from_indices(batch_indices)
# plt.hist(metamat[0].flatten())

# n_pixels = np.prod(metamat[0].shape[:2])
# # plt.imshow(gray_box(metamat[0], 25,25,50) / 255)
# # plt.show()
# step= 5
# saliency_map = np.zeros([224/step + 1,224/step + 1])
# locations = []
# images = []
#
# %matplotlib inline
#
# pad = 25/2
# for row in range(pad,224 - 2 - pad, step):
#     for col in range(pad,224 - 2 - pad, step):
#         locations.append((row / step, col / step))
#         # print locations[-1]
#         images.append(gray_box(metamat[0], row, col,25))
#         # plt.imshow(images[-1])
#         # plt.show()
#
# batch_size = 50
# for i in range(0, len(locations), batch_size):
#     image_batch = np.array(images[i:i+batch_size])
#     for image in image_batch:
#         plt.imshow(image)
#         plt.show()
#         time.sleep(.1)
#
#     extra_features_batch = np.array([extra_features[0]] * len(image_batch))
#     # extra_features_batch = np.array((list(extra_features)*2)[:len(image_batch)])
#     # print extra_features_batch.shape
#     # break
#     p_damages = model.sess.run(model.pred_probabilities,
#                        feed_dict={model.heightmap_ph: image_batch,
#                                   model.extra_features_ph:extra_features_batch,
#                                   model.keep_prob_ph: 1})
#     # plt.hist(np.array(image_batch).flatten())
#     # plt.show()
#     print p_damages
#     print np.mean(image_batch[0])
#     for prob, loc in zip(p_damages, locations[i: i + batch_size]):
#         row, col = loc
#         saliency_map[row, col] = prob
#     break
#
# plt.imshow(metamat[0] / 255)
# plt.show()
# # base_p_damage = model.sess.run([loss, accuracy, pred_probabilities],
#                 #    feed_dict={vgg.imgs: [X[0]], labels: Y, wind_ph: wind, keep_prob_ph:1.0}) [2][0][0]
# plt.imshow(saliency_map[:-1, :-1], cmap = 'viridis')
# plt.colorbar()
# plt.show()


#########
batch_indices = [349829]
metamat, extra_features, y_true = data_handler.get_data_batch_from_indices(batch_indices)
pad = 25/2
step= 5
box_size = 25
locations = []
images = []

for row in range(pad,224 - 2 - pad, step):
    for col in range(pad,224 - 2 - pad, step):
        locations.append((row / step, col / step))
        images.append(gray_box(metamat[0], row, col,box_size))
        # plt.imshow(images[-1])
        # plt.show()


def print_probs(images, extra_features):
    pred_probabilities = model.sess.run(model.pred_probabilities,
        feed_dict={model.heightmap_ph:images,
        model.extra_features_ph:extra_features,
        model.keep_prob_ph: 1})
    # print "===="
    # plt.imshow(images[0])
    # plt.show()
    # print pred_probabilities[:,0 ]
    # print np.std(pred_probabilities[:,0 ])
    # print hash(tuple(pred_probabilities.flatten()))
    return pred_probabilities


probabilities = []
batch_size = 10
for i in range(0, len(locations), batch_size):
    image_batch = np.array(images[i: i+batch_size])
    # plt.imshow(image_batch[0])
    # plt.show()
    extra_features_batch = np.array([extra_features[0]] * batch_size)
    print image_batch.shape, extra_features_batch.shape
    print extra_features_batch.shape, image_batch.shape
    probs = print_probs(image_batch, extra_features_batch)
    probabilities.extend(probs)
probabilities = np.array(probabilities)[:,1]
probabilities.shape
probabilities = probabilities.reshape([int(np.sqrt(len(probabilities)))]*2)
print probabilities
plt.imshow(probabilities, cmap='viridis')
plt.show()
plt.imshow(images[0])
plt.show()
#
print np.array(images).shape

random_like = lambda x: np.random.normal(loc=0.0, scale=1.0, size=x.shape)
