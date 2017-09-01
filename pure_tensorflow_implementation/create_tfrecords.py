PYSOLR_PATH = '/home/isaac/Desktop/storm_damage_prediction/pure_tensorflow_implementation'
from matplotlib import pyplot as plt
%matplotlib inline
import sys
if not PYSOLR_PATH in sys.path:
    sys.path.append(PYSOLR_PATH)

import tensorflow as tf
import data_loading
reload(data_loading)
data_handler = data_loading.DataHandler()


for index in range(100)
    lon = [data_handler.meta['lon'][index]]
    lat = [data_handler.meta['lat'][index]]
    heightmap = data_handler.get_heightmap_around_lat_lon(lat, lon, window_width_pixels=126)


plt.imshow(heightmap, cmap='viridis')
