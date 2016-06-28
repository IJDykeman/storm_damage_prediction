
# from tsne import bh_sne # thi is the correct tsne to use.  It's the one discussed btnw
# from bhtsne import bh_tsne
import sklearn.manifold
from matplotlib import pyplot as plt
import pandas
import scipy
import numpy as np
import os
import gc
from time import gmtime, strftime
import seaborn as sns
from os import listdir
from os.path import isfile, join
import math
from scipy.stats.stats import pearsonr
import random as rand
from sklearn.preprocessing import normalize
from collections import defaultdict
def memo(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper

def zero_to_one(array):
    # array = array - np.min(array)
    # array = 2*(array/np.max(array))-1
    # return np.nan_to_num(array)


    scaler = sklearn.preprocessing.MinMaxScaler(feature_range = (0,1))
    array[array == np.inf] = 0
    array[array == -np.inf] = 0
    array[array == np.nan] = 0
    array = array.fillna(0)
    result = scaler.fit_transform(array)
    # print(min(result))
    return result


# @memo
def load_dataset(path):
    print("loading...")
    gc.collect() # collect garbage
    data = pandas.read_hdf(path, '/df')
    df = pandas.DataFrame(data)
    data_dict = {}
    for label in set(df._get_numeric_data().columns).union({'hcad'}):
        # union hcad to ensure that hcad col comes in even if not considered numerical
        # if label != 'hcad':
        data_dict[label] = df[label].astype(float)
        data_dict[label] = zero_to_one(data_dict[label])
        # df[label][df[label] > 1] = 1.0

    # df['hcad'] = df['hcad'].astype(float)
    result = pandas.DataFrame.from_dict(data_dict)

    result = result.replace([np.inf, -np.inf], 1)
    
    return result.sort(['hcad']).fillna(0)
    
def fast_tsne(df_data, dest_folder="", n = None, file_tag= "", embedded_dimensions=2, perplexity = 50):

    df_data = df_data.drop('hcad', 1) # don't embed the hcad number!
    df_data = np.array(df_data)[:n]
    embedding = bh_sne(np.array(df_data)[:n], perplexity=perplexity, d = embedded_dimensions)

    # result_2d = {}
    # result_2d['hcad'] = df_data['hcad'][:n]
    # result_2d['x'] = zero_to_one(embedding[:, 0])
    # result_2d['y'] = zero_to_one(embedding[:, 1])
    # result_2d = pandas.DataFrame.from_dict(result_2d)
    #name = file_tag+"_"+"_".join(df_data.columns)[:40] + "_n:"+str(len(result))
    #result.to_pickle(dest_folder+name)
    return embedding


def slow_tsne(df_data, dest_folder, n = None, file_tag= "", embedded_dimensions=2, perplexity = 50):
    result_2d = {}
    result_2d['hcad'] = df_data['hcad'][:n]
    df_data = df_data.drop('hcad', 1) # don't embed the hcad number!
    df_data = np.array(df_data)[:n]
    embedding = np.array(list(bh_tsne(np.array(df_data)[:n], perplexity=perplexity)))
    print(embedding)

    result_2d['x'] = zero_to_one(embedding[:, 0])
    result_2d['y'] = zero_to_one(embedding[:, 1])
    result_2d = pandas.DataFrame.from_dict(result_2d)
    #name = file_tag+"_"+"_".join(df_data.columns)[:40] + "_n:"+str(len(result))
    #result.to_pickle(dest_folder+name)
    return embedding

def hist_2d(vis_x,vis_y):
    hh, locx, locy = scipy.histogram2d(vis_x, vis_y, bins=[200,200])
    fig = plt.figure(frameon=False)
    fig.set_size_inches(30,30)
    plt.imshow(np.flipud(hh.T),cmap='jet', interpolation='none', shape = (1,1))
    plt.colorbar()
    


def pairwise_plot(pddf, sqrt = False):
    if sqrt:
        pddf = np.sqrt(pddf)
    axes = pandas.tools.plotting.scatter_matrix(pddf, alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    

def old_fast_show_ratio_plot(xy_points, y_data, log = False, normalize_buckets=True):
    if log:
        y_data = np.log(y_data)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,3)
    plt.hist(y_data)
    plt.show()

    buckets = defaultdict(list)
    resolution = 200
    x = np.array(xy_points['x'])
    y = np.array(xy_points['y'])
    H, xedges, yedges = numpy.histogram2d(x,y, bins=resolution, weights = y_data)
    H_nums, dummy2, dummy1 = numpy.histogram2d(x,y, bins=resolution)
    plt.show()
    fig = plt.figure(frameon=False)
    fig.set_size_inches(12,12)
    if normalize_buckets:
        H=H/H_nums
    H[H_nums == 0.0] = numpy.nan
#     if log:
#         H = np.log(H)
    

    plt.imshow(H, 
               interpolation='nearest', cmap=cm.gist_rainbow)
    plt.colorbar()
    plt.show()
    return np.nan_to_num(H)

#===============================================================================

def colored_scatter(xy_points, y_data):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(20,20)
        plt.scatter(xy_points['x'], xy_points['y'], c=y_data,  marker='x', facecolor='b', cmap='jet')
        plt.colorbar()
        plt.show()

def shuffle_in_unison(a, b):
    assert a.shape[0] == b.shape[0]
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)
        

def train_test_split(x_mat_in, y_col_in):
    x_mat = x_mat_in.copy()
    y_col = y_col_in.copy()

    shuffle_in_unison(x_mat, y_col)

    split1 = len(y_col)*10/100
    split2 = len(y_col)*20/100


    X_test = np.expand_dims(np.expand_dims(x_mat[:split1], axis=1), axis=3)
    y_test = y_col[:split1] # limit training data amount, as opposed to 600000
    X_val = np.expand_dims(np.expand_dims(x_mat[split1:split2], axis=1), axis=3)
    y_val = y_col[split1:split2]
    X_train = np.expand_dims(np.expand_dims(x_mat[split2:], axis=1), axis=3)
    y_train = y_col[split2:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_mega_hcad():

    mega_hcad = {}

    for column in hcad:
        # for index, dataset in enumerate(hcad_data):
            # mega_hcad[column+"_hcad_"+str(index)] = dataset[column]
        mega_hcad[column+"_hcad_"] = hcad[column]

    # for column in WIND:
    #     if not 'hist' in column:
    #         mega_hcad[column+"_wind"] = WIND[column]


    mega_hcad = pandas.DataFrame.from_dict(mega_hcad).as_matrix()


    y_data_np = Y_DATA.as_matrix()
    shuffle_in_unison(mega_hcad, y_data_np)
    
    y_column = 6


    # mega_hcad_nonzero = mega_hcad[y_data_np[:,y_column]!=0]
    # y_data_np_nonzero = y_data_np[y_data_np[:, y_column]!=0]
    # mega_hcad_zero = mega_hcad[y_data_np[:,y_column]==0][:250000]
    # y_data_np_zero = y_data_np[y_data_np[:,y_column]==0][:250000]
    
    # mega_hcad = np.concatenate((mega_hcad_nonzero, mega_hcad_zero), axis=0)
    # y_data_np = np.concatenate((y_data_np_nonzero, y_data_np_zero), axis=0)

    
    print("hcad length", (mega_hcad.shape))
    print("y_data length", (y_data_np.shape))


    return train_test_split(mega_hcad, y_data_np[:, y_column])


print ("helpers executed")