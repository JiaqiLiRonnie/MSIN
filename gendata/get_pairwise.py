import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap


def make_positon(dataMat, X_Y, i):

	frame_number = int(len(dataMat)/3)
	joint_map = np.zeros((9,300,25,2))
	zheng = int(300/frame_number)
	yu = int(300%frame_number)

	for n in range(frame_number):
		box_center_1 = ((dataMat[int(3*n)][0] + dataMat[int(3*n)][2])/2, (dataMat[int(3*n)][1] + dataMat[int(3*n)][3])/2)
		box_center_2 = ((dataMat[int(3*n+1)][0] + dataMat[int(3*n+1)][2])/2, (dataMat[int(3*n+1)][1] + dataMat[int(3*n+1)][3])/2)
		box_center_3 = ((dataMat[int(3*n+2)][0] + dataMat[int(3*n+2)][2])/2, (dataMat[int(3*n+2)][1] + dataMat[int(3*n+2)][3])/2)		 
		for joint in range(25):
			if not np.any(np.isnan(X_Y[i, :, n, joint, :])):
				joint_map[0,n,joint,0] = box_center_1[0]-X_Y[i, 0, n, joint, 0]
				joint_map[1,n,joint,0] = box_center_1[1]-X_Y[i, 1, n, joint, 0]
				joint_map[2,n,joint,0] = X_Y[i, 2, int(n), joint, 0]
				joint_map[3,n,joint,0] = box_center_2[0]-X_Y[i, 0, n, joint, 0]
				joint_map[4,n,joint,0] = box_center_2[1]-X_Y[i, 1, n, joint, 0]
				joint_map[5,n,joint,0] = X_Y[i, 2, int(n), joint, 0]
				joint_map[6,n,joint,0] = box_center_3[0]-X_Y[i, 0, n, joint, 0]
				joint_map[7,n,joint,0] = box_center_3[1]-X_Y[i, 1, n, joint, 0]
				joint_map[8,n,joint,0] = X_Y[i, 2, int(n), joint, 0]
				if X_Y[i, 0, int(n), joint, 1] != 0:
					joint_map[0,n,joint,1] = box_center_1[0]-X_Y[i, 0, n, joint, 1]
					joint_map[1,n,joint,1] = box_center_1[1]-X_Y[i, 1, n, joint, 1]
					joint_map[2,n,joint,1] = X_Y[i, 2, int(n), joint, 1]
					joint_map[3,n,joint,1] = box_center_2[0]-X_Y[i, 0, n, joint, 1]
					joint_map[4,n,joint,1] = box_center_2[1]-X_Y[i, 1, n, joint, 1]
					joint_map[5,n,joint,1] = X_Y[i, 2, int(n), joint, 1]
					joint_map[6,n,joint,1] = box_center_3[0]-X_Y[i, 0, n, joint, 1]
					joint_map[7,n,joint,1] = box_center_3[1]-X_Y[i, 1, n, joint, 1]
					joint_map[8,n,joint,1] = X_Y[i, 2, int(n), joint, 1]
			else:
				joint_map[0,n,joint,0] = box_center_1[0]
				joint_map[1,n,joint,0] = box_center_1[1]
				joint_map[2,n,joint,0] = 0.0
				joint_map[3,n,joint,0] = box_center_2[0]
				joint_map[4,n,joint,0] = box_center_2[1]
				joint_map[5,n,joint,0] = 0.0
				joint_map[6,n,joint,0] = box_center_3[0]
				joint_map[7,n,joint,0] = box_center_3[1]
				joint_map[8,n,joint,0] = 0.0
				
				if X_Y[i, 0, int(n), joint, 1] != 0:
					joint_map[0,n,joint,1] = box_center_1[0]
					joint_map[1,n,joint,1] = box_center_1[1]
					joint_map[2,n,joint,1] = 0.0
					joint_map[3,n,joint,1] = box_center_2[0]
					joint_map[4,n,joint,1] = box_center_2[1]
					joint_map[5,n,joint,1] = 0.0
					joint_map[6,n,joint,1] = box_center_3[0]
					joint_map[7,n,joint,1] = box_center_3[1]
					joint_map[8,n,joint,1] = 0.0

	for m in range(zheng-1):
		joint_map[:,(m+1)*frame_number:(m+2)*frame_number,:,:] = joint_map[:,0:frame_number,:,:]
	joint_map[:,zheng*frame_number:,:,:] = joint_map[:,0:yu,:,:]

	return joint_map 
	
	
if __name__ == '__main__':
	position_root = ""
	X_Y = np.load("")
	f = open(r"",'rb')
	inf = pickle.load(f)
	fp = open_memmap(
        '',
        dtype='float32',
        mode='w+',
        shape=(len(inf[0]), 9, 300, 25, 2))
	i = 0
	for skeleton in inf[0]: 
		print(skeleton)
		for classes in os.listdir(position_root): #
			if classes == skeleton[16:20]:
				for txt in os.listdir(os.path.join(position_root,classes)):
					if txt[0:20] == skeleton[0:20]:
						dataMat = []
						f = open(os.path.join(os.path.join(position_root,classes),txt))
						box_data = f.readlines()
						for line in box_data:
							curLine=line.strip().split(",")
							floatLine=list(map(float,curLine))
							dataMat.append(floatLine)
						position = make_positon(dataMat, X_Y, i) 
						fp[i, :, :, :, :] = position
						print(i)
		i = i + 1