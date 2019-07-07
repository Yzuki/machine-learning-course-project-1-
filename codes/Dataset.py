import numpy as np
import math
import cv2

positive_train = np.load('positive_train_hog.npy')
positive_test = np.load('positive_test_hog.npy')
negative_train = np.load('negative_train_hog.npy')
negative_test = np.load('negative_test_hog.npy')

label_positive_train = np.ones((len(positive_train),1),dtype = 'int32')
label_positive_test = np.ones((len(positive_test),1),dtype = 'int32')
label_negative_train = -np.ones((len(negative_train),1),dtype = 'int32')
label_negative_test = -np.ones((len(negative_test),1),dtype = 'int32')


train_x = np.concatenate((positive_train,negative_train),axis = 0)
train_y = np.concatenate((label_positive_train,label_negative_train),axis = 0)

test_x = np.concatenate((positive_test,negative_test),axis = 0)
test_y = np.concatenate((label_positive_test,label_negative_test),axis = 0)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
np.save('train_x.npy',train_x)
np.save('train_y.npy',train_y)
np.save('test_x.npy',test_x)
np.save('test_y.npy',test_y)