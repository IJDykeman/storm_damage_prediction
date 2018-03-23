PYSOLR_PATH = '/home/isaac/Desktop/storm_damage_prediction/pure_tensorflow_implementation'
# %matplotlib inline

import sys
if not PYSOLR_PATH in sys.path:
    sys.path.append(PYSOLR_PATH)

import model as model_module
import data_loading
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pyproj

def summarize(name, value, global_step, summary_writer):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    summary_writer.add_summary(summary, global_step)
# data_handler = data_loading.DataHandler()
# def get_batch(batch_size, is_validation=False):
#     # return self.get_data_batch_from_indices(self.get_example_indices(batch_size, is_validation))
#     tuples = data_handler.pool.map(data_handler.get_data_batch_from_indices, data_handler.get_example_indices(batch_size, is_validation))
#     metamat = np.array([x[0] for x in tuples])
#     wind_speed = np.array([x[1] for x in tuples])
#     wind_dir = np.array([x[2] for x in tuples])
#     categorical_y = np.array([x[3] for x in tuples])
    
# print "getting batch..."
# get_batch(32)
# quit()
BATCH_SIZE = 64

def get_trained_model(tag = 'model1', verbose=False, remove_image = False, remove_wind = False, remove_hcad = False):
    model = model_module.Model(tag = tag, verbose=verbose, remove_image=remove_image, remove_wind=remove_wind, remove_hcad=remove_hcad)
    data_handler = data_loading.DataHandler()
    if verbose: print data_handler.hcad.shape
    if verbose: print data_handler.wind_data.shape
    i=0
    print "    starting training"
    for _ in range(1500):
        metamat, wind_speed, wind_dir, hcad, class_y = data_handler.get_batch(BATCH_SIZE, is_validation=False)

        _,loss_val,_, summary_val = \
            model.sess.run([model.optimizer, model.loss, model.accuracy, model.merged_summaries],
            feed_dict={model.heightmap_ph:metamat,
            # model.extra_features_ph:extra_features,
            model.wind_speed_placeholder: wind_speed,
            model.wind_direction_placeholder: wind_dir,
            model.hcad_placeholder: hcad,
            model.labels_ph:class_y,
            model.keep_prob_ph: .7})
        if i % 5 == 0:
            model.train_summary_writer.add_summary(summary_val, i)
            metamat, wind_speed, wind_dir, hcad, class_y = data_handler.get_batch(BATCH_SIZE, is_validation=True)

            loss_val = \
                model.sess.run([model.loss],
                feed_dict={model.heightmap_ph:metamat,
                # model.extra_features_ph:extra_features,
                model.wind_speed_placeholder: wind_speed,
                model.wind_direction_placeholder: wind_dir,
                model.hcad_placeholder: hcad,
                model.labels_ph:class_y,
                model.keep_prob_ph: 1})
            summarize("validation_loss", loss_val[0],i,model.train_summary_writer)

        #if i % 50 == 0:
        #    model.save(i)
        i += 1
    print "    training complete"
    return model


if __name__ == "__main__":
    import eval_auc
    model = get_trained_model(tag='adam_01', remove_image = True, remove_wind = True, remove_hcad = True)
    eval_auc.evaluate_auc(model)
    tf.reset_default_graph()
