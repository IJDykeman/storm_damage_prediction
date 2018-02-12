import numpy as np
from matplotlib import pyplot as plt

def gray_box(image, row, col, size):
    result = np.copy(image)
    result[row - size/2 : row + size/2, col - size/2 : col + size/2] = np.max(image)
    #     result = scipy.interpolate.interp2d(x, y, z, kind='linear', copy=True, bounds_error=False, fill_value=np.nan)
    return result

def get_saliency_map (image, extra_features_vec, box_size, step, model, show=False):
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
    if show:
        plt.imshow(probabilities, cmap='viridis')
        plt.show()
    return probabilities
