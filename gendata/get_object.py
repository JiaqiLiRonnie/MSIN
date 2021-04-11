import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

def make_positon(feature_data,frame_data_i):

	for m in range(300):
		if frame_data_i[m] == 0 :
			break
	if m != 0:
		feature_map = np.zeros((256,768))
		cishu = int(256/m)
		yu = int(256%m)
		feature_map[:,0:3*m] = feature_data[:,0:3*m]
		for ci in range(cishu-1):
			feature_map[:,3*m*(ci+1):3*m*(ci+2)] = feature_data[:,0:3*m]
		feature_map[:,3*m*cishu:768] = feature_data[:,0:(768-3*m*cishu)]
	else:
		feature_map = np.zeros((256,768))
		for b in range(768):
			feature_map[:,b] = feature_data[:,b]

	return feature_map 


if __name__ == '__main__':
	position_root = ""
	
	f = open(r"",'rb')
	inf = pickle.load(f)
	
	frame_data = np.load("")
	
	fp = open_memmap(
        '',
        dtype='float32',
        mode='w+',
        shape=(len(inf[0]), 2, 256, 768))

	i = 0
	for skeleton in inf[0]: 
		print(skeleton)
		for classes in os.listdir(position_root): 	
			if classes == skeleton[16:20]:
				for npy in os.listdir(os.path.join(position_root,classes)): # /data/shm/feature/A010
					if npy[0:20] == skeleton[0:20]:
						feature_data = np.load(os.path.join(os.path.join(position_root,classes),npy))                       
						frame_data_i = frame_data[i,0,:,0,0]
						feature = make_positon(feature_data, frame_data_i)
						fp[i, 0, :, :] = feature
						print(i)
		i = i + 1