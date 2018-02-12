# this is to make the environment compatible with the Atom Hydrogen
# interactive editor package.  You can ignore this code if that doesn't mean
# anything to you.
PYSOLR_PATH = '/home/isaac/Desktop/storm_damage_prediction/pure_tensorflow_implementation'
import sys
if not PYSOLR_PATH in sys.path:
    sys.path.append(PYSOLR_PATH)
#########

import data_loading
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import sklearn.manifold
print sklearn.metrics.roc_auc_score

data_handler = data_loading.DataHandler()

print "wind data means"
print np.mean(data_handler.wind_data[data_handler.train_indices_randomized], axis=0)
print "wind data mins"
print np.max(data_handler.wind_data[data_handler.train_indices_randomized], axis=0)
print "wind data maxes"
print np.min(data_handler.wind_data[data_handler.train_indices_randomized], axis=0)
print "========"
print "hcad means"
print np.mean(data_handler.hcad[data_handler.train_indices_randomized], axis=0)
print "hcad maxes"
print np.max(data_handler.hcad[data_handler.train_indices_randomized], axis=0)
print "hcad mins"
print np.min(data_handler.hcad[data_handler.train_indices_randomized], axis=0)

