
# coding: utf-8

# In[ ]:

import pylab
pylab.rcParams['figure.figsize'] = (8.0, 8.0)
from tsne import bh_sne
from matplotlib import pyplot as plt
import pandas
import scipy
import numpy as np
import sklearn.manifold
import os
import gc
from time import gmtime, strftime
import seaborn as sns


# In[ ]:

def load_data(path):
    gc.collect()
    data = pandas.read_hdf(path, '/df')
    np_data = np.array(data).astype(float)
    del(data)
    data=None
    np_data = np_data / np_data.max(axis=0)
    
    np.random.shuffle(np_data)
    np.nan_to_num(np_data)
    np_data[np.isnan(np_data)] = 0
    return np_data


# In[ ]:

def visualize(muh_data, buckets = 100, show=False, size=40, path=None, bandwidth=2):
    vis_x = muh_data[:, 0]
    vis_y = muh_data[:, 1]


    #histogram definition
    xyrange = [[-size,size],[-size,size]] # data range
    bins = [buckets,buckets] # number of bins
    #thresh = 3  #density threshold

    #data definition


    # histogram the data
    #hh, locx, locy = scipy.histogram2d(vis_x, vis_y, range=xyrange, bins=bins)


    #plt.imshow(np.flipud(hh.T),cmap='jet',extent=np.array(xyrange).flatten(), interpolation='none', shape = (1000,1000))
    #plt.colorbar()
    if(show):
        #
        #sns.set(color_codes=True)
        sns.kdeplot(vis_x, vis_y, shade=True,  n_levels=20, bw=bandwidth)
        plt.show()
    else:
        assert(path != None)
        save_path = folder+"".join(path.split('/')[-2:])
        save_path = save_path+"_buckets:"+str(buckets)
        save_path = save_path  +".png"
        plt.savefig(save_path)
    #plt.clf()

def save(muh_data, folder, tag=""):
    save_path = folder+"np_array_"+tag
    print "saving to",save_path
    np.save(save_path,muh_data)


# In[ ]:

def process_full():
    paths = ["/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df.hd",
             "/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df_100.hd",
             "/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df_200.hd",
             "/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df_400.hd",
             "/home/isaac/Dropbox/data_for_brian/hcad_features/hcad_df_1000.hd",
             "/home/isaac/Dropbox/data_for_brian/terrain_features/dsmgrid/terrain_100.hd",
             "/home/isaac/Dropbox/data_for_brian/terrain_features/dsmgrid/terrain_200.hd",
             "/home/isaac/Dropbox/data_for_brian/terrain_features/dsmgrid/terrain_400.hd",
             "/home/isaac/Dropbox/data_for_brian/terrain_features/dsmgrid/terrain_1000.hd",
             "/home/isaac/Dropbox/data_for_brian/terrain_features/dtmgrid/terrain_100.hd",
             "/home/isaac/Dropbox/data_for_brian/terrain_features/dtmgrid/terrain_200.hd",
             "/home/isaac/Dropbox/data_for_brian/terrain_features/dtmgrid/terrain_400.hd",
             "/home/isaac/Dropbox/data_for_brian/terrain_features/dtmgrid/terrain_1000.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad100_hist8x8.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad100_hist16x16.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad100_hist36x36.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad200_hist8x8.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad200_hist16x16.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad200_hist36x36.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad400_hist8x8.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad400_hist16x16.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad400_hist36x36.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad1000_hist8x8.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad1000_hist16x16.mat.hd",
             #"/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad1000_hist36x36.mat.hd"
            ]

    folder = "./autorun9/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    print "started at "+strftime("%Y-%m-%d %H:%M:%S", gmtime())
    vis_data = None
    for path in paths:
        np_data = load_data(path)
        print np_data
        for n in [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
            vis_data = bh_sne(np_data[:n], perplexity=30)
            save(vis_data, folder, tag = str(n)+"_"+path.split('/')[-1].split('.')[0])

            del(vis_data)
            vis_data=None
            print "done with "+path+" at "+strftime("%Y-%m-%d %H:%M:%S", gmtime())
        del(np_data)
        np_data=None
            
process_full()


# In[ ]:

loaded_data = np.load("./autorun8_hcad_df_various_sample_numbers/numpy_array_of_sne_points1000.npy")


# In[ ]:

visualize(loaded_data[:], show=True, size=45, bandwidth=.5)
visualize(loaded_data[:500], show=True, size=45, bandwidth=.5)
#visualize(loaded_data, show=True, size=45, bandwidth=1)
#visualize(loaded_data, show=True, size=45, bandwidth= 2)


# In[ ]:




# In[ ]:



