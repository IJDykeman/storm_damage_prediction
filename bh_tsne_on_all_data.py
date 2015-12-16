
# coding: utf-8

# In[ ]:

from tsne import bh_sne
from matplotlib import pyplot as plt
import pandas
import scipy
import numpy as np


# In[ ]:

def visualize(muh_data, buckets = 100):
    vis_x = muh_data[:, 0]
    vis_y = muh_data[:, 1]


    #histogram definition
    xyrange = [[-15,15],[-15,15]] # data range
    bins = [buckets,buckets] # number of bins
    #thresh = 3  #density threshold

    #data definition
    N = len(vis_x);

    # histogram the data
    hh, locx, locy = scipy.histogram2d(vis_x, vis_y, range=xyrange, bins=bins)


    plt.imshow(np.flipud(hh.T),cmap='jet',extent=np.array(xyrange).flatten(), interpolation='none', shape = (1000,1000))
    plt.colorbar()   
    #plt.show()
    save_path = folder+"".join(path.split('/')[-2:])
    save_path = save_path+"_buckets:"+str(buckets)
    save_path = save_path  +".png"
    plt.savefig(save_path)
    plt.clf()

def save(muh_data):
    np.save(folder+"".join(path.split('/')[-2:])+"_numpy_array_of_sne_points",vis_data)


# In[ ]:

paths = [#"/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df.hd",
         #"/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df_100.hd",
         #"/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df_200.hd",
         #"/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df_400.hd",
         #"/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df_1000.hd",
         #"/home/isaac/Dropbox/data_for_brian/terrain_features/dsmgrid/terrain_100.hd",
         #"/home/isaac/Dropbox/data_for_brian/terrain_features/dsmgrid/terrain_200.hd",
         #"/home/isaac/Dropbox/data_for_brian/terrain_features/dsmgrid/terrain_400.hd",
         #"/home/isaac/Dropbox/data_for_brian/terrain_features/dsmgrid/terrain_1000.hd",
         #"/home/isaac/Dropbox/data_for_brian/terrain_features/dtmgrid/terrain_100.hd",
         #"/home/isaac/Dropbox/data_for_brian/terrain_features/dtmgrid/terrain_200.hd",
         #"/home/isaac/Dropbox/data_for_brian/terrain_features/dtmgrid/terrain_400.hd",
         #"/home/isaac/Dropbox/data_for_brian/terrain_features/dtmgrid/terrain_1000.hd",
         #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad100_hist8x8.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad100_hist16x16.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad100_hist36x36.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad200_hist8x8.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad200_hist16x16.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad200_hist36x36.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad400_hist8x8.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad400_hist16x16.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad400_hist36x36.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad1000_hist8x8.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad1000_hist16x16.mat.hd",
         "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad1000_hist36x36.mat.hd"
        ]
import os
import gc
folder = "./autorun4/"
if not os.path.exists(folder):
    os.makedirs(folder)
path = ""
from time import gmtime, strftime
print "started at "+strftime("%Y-%m-%d %H:%M:%S", gmtime())
for muh_path in paths:
    gc.collect()
    path = muh_path
    data = pandas.read_hdf(path, '/df')
    np_data = np.array(data).astype(float)
    del(data)
    data=None
    vis_data = bh_sne(np_data)
    visualize(vis_data, buckets = 100)
    visualize(vis_data, buckets = 200)
    visualize(vis_data, buckets = 400)
    save(vis_data)
    del(np_data)
    np_data=None
    del(vis_data)
    vis_data=None
    print "done with "+path+" at "+strftime("%Y-%m-%d %H:%M:%S", gmtime())


# In[ ]:



