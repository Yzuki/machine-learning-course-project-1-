import numpy as np
import math
import cv2
import random
import os
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
import gc

img = cv2.imread('img.jpg')
print(img.shape)
def enumerate(img):
	len1 = img.shape[0]
	len2 = img.shape[1]
	imgs = []
	records = []
	size1 = 150
	size2 = 250
	for i in range(80,len1,40):
		for j in range(40,len2,40):
					if(i+size1<=len1 and j+size2<=len2):
						imgs.append(cv2.resize(img[i:i+size1,j:j+size2], (96, 96)))	
						records.append([i,j,size1,size2])
	return imgs,records

imgs,records = enumerate(img)
imgs_hog = []
from skimage import feature
for i in imgs:
	out,hog = feature.hog(
			i,orientations=9,
			pixels_per_cell = (16,16),
			cells_per_block = (2,2),
			visualize = True,
			feature_vector = True,
			multichannel = True)
	imgs_hog.append(out)
imgs_hog = np.array(imgs_hog)

img1 = img
logistic_theta = np.load('param_logistic.npy')
pre_y = 1/(1+np.exp(-np.matmul(imgs_hog,logistic_theta)))
pre_y = np.where(pre_y>=0.5,1,-1)
print(pre_y)
pre_y = pre_y.astype('int32')

for i in range(pre_y.shape[0]):
	if(pre_y[i][0] == 1):
		cv2.rectangle(img1, (records[i][0],records[i][1]), (records[i][0]+records[i][2],records[i][1]+records[i][3]),(0,0,0),2)
cv2.imshow('logistic',img1)

img3 = img
from sklearn.externals import joblib
linear_svm = joblib.load('linear_svm.m')
predict = linear_svm.predict(imgs_hog)
for i in range(pre_y.shape[0]):
	if(pre_y[i][0] == 1):
		cv2.rectangle(img3, (records[i][0],records[i][1]), (records[i][0]+records[i][2],records[i][1]+records[i][3]),(0,0,0),2)
cv2.imshow('linear_svm',img3)

img4 = img
from sklearn.externals import joblib
rbf_svm = joblib.load('rbf_svm.m')
predict = rbf_svm.predict(imgs_hog)
for i in range(pre_y.shape[0]):
	if(pre_y[i][0] == 1):
		cv2.rectangle(img4, (records[i][0],records[i][1]), (records[i][0]+records[i][2],records[i][1]+records[i][3]),(0,0,0),2)
cv2.imshow('rbf_svm',img4)

from keras.models import load_model
m = load_model("cnn.hdf5")
img2 = img
predict = m.predict(np.array(imgs))
for i in range(predict.shape[0]):
	if(predict[i][1] >= 0.5):
		cv2.rectangle(img2, (records[i][0],records[i][1]), (records[i][0]+records[i][2],records[i][1]+records[i][3]),(0,0,0),2)
cv2.imshow('cnn',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()






