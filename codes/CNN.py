import numpy as np
import math
import cv2
import random
import os
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
import gc
import matplotlib.pyplot as plt
import matplotlib
p_train_img = []
for i in os.listdir('positive_train'):
	img = cv2.imread('positive_train/'+ i)
	p_train_img.append(img)
n_train_img = []
for i in os.listdir('negative_train'):
	img = cv2.imread('negative_train/'+ i)
	n_train_img.append(img)
p_test_img = []
for i in os.listdir('positive_test'):
	img = cv2.imread('positive_test/'+ i)
	p_test_img.append(img)
n_test_img = []
for i in os.listdir('negative_test'):
	img = cv2.imread('negative_test/'+ i)
	n_test_img.append(img)
train_x = np.array(p_train_img+n_train_img)
train_y = np.load('train_y.npy').reshape(-1,1)
train_y = to_categorical(np.where(train_y>0,1,0))
del p_train_img,n_train_img
gc.collect()
test_x = np.array(p_test_img+n_test_img)
test_yy = np.load('test_y.npy').reshape(-1,1)
test_y = to_categorical(np.where(test_yy>0,1,0))
del p_test_img,n_test_img
gc.collect()

def CNN():
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation = 'relu',input_shape=(96, 96, 3)))
	model.add(BatchNormalization())
	model.add(MaxPool2D((2, 2)))
	
	model.add(Conv2D(64, (3, 3),activation = 'relu'))
	model.add(BatchNormalization())
	model.add(MaxPool2D((2, 2)))

	model.add(Conv2D(64, (3, 3),activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(64, activation='relu',name = 'dense'))
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
	return model

m = CNN()
print(m.summary())
from keras import callbacks
checkpoint = callbacks.ModelCheckpoint('cnn.hdf5', monitor='val_acc', save_best_only=True)
early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=3)
m.fit(train_x,train_y,shuffle = True,validation_data = (test_x,test_y), batch_size=32, epochs=1000,callbacks = [checkpoint,early_stopping])
from keras.models import load_model
model = load_model("cnn.hdf5")
final_loss, final_acc = model.evaluate(test_x,test_y, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))


