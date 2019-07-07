import numpy as np
import math
import cv2
import random
import sklearn
from sklearn import svm
train_x = np.load('train_x.npy').reshape(-1,900)
train_y = np.load('train_y.npy').reshape(-1,1)
test_x = np.load('test_x.npy').reshape(-1,900)
test_y = np.load('test_y.npy').reshape(-1,1)


class SVM:
	def __init__(self):
		self.model = None
		pass
	def linear(self,x,y):
		self.model = sklearn.svm.SVC(kernel = 'linear')
		self.model.fit(x,y.ravel())
	def rbf(self,x,y):
		self.model = sklearn.svm.SVC(kernel = 'rbf')
		self.model.fit(x,y.ravel())
	def sigmoid(self,x,y):
		self.model = sklearn.svm.SVC(kernel = 'sigmoid')
		self.model.fit(x,y.ravel())
	def support_vectors(self):
		sv_list = self.model.support_
		return sv_list
	def predict(self,x):
		pre_y = self.model.predict(x)
		return pre_y

m = SVM()
m1 = SVM()
m2 = SVM()
m.linear(train_x,train_y)
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
joblib.dump(m.model,'linear_svm.m')
f = open('train_names.txt')  #visualize support vectors
train_names = f.readlines()
f.close()



m1.rbf(train_x,train_y)
m2.sigmoid(train_x,train_y)
joblib.dump(m1.model,'rbf_svm.m')

print("Linear SVM acc:",accuracy_score(m.predict(test_x),test_y))
print("RBF acc:",accuracy_score(m1.predict(test_x),test_y))
print("Sigmoid acc:",accuracy_score(m2.predict(test_x),test_y))

sv_idx = list(m.support_vectors())
sv = []
for i in sv_idx:
	sv.append(train_names[i].strip())
sv = random.sample(sv,3)
print("support vectors:", sv)
sv_img = []
for i in sv:
	positive = 'positive_train/'+i
	negative = 'negative_train/'+i
	if(i.count('_') == 1):
		p = cv2.imread(positive)
		sv_img.append(p)
	else:
		n = cv2.imread(negative)
		sv_img.append(n)
for i in range(len(sv_img)):
	cv2.imshow(sv[i],sv_img[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
