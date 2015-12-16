from tsne import bh_sne
from matplotlib import pyplot as plt
import pandas
import scipy.io as sio

path = "/home/isaac/Dropbox/data_for_brian/wind_features/hcad_interp_withoutpartial_rad400_hist8x8.mat.hd"
#data = sio.loadmat(path)
print pandas.read_hdf(path, '/df')
def print_keys():
	print pandas.HDFStore(path).keys()