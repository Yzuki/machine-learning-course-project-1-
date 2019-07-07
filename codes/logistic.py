import numpy as np
import math
import cv2
import random
import sklearn
train_x = np.load('train_x.npy').reshape(-1,900)
train_y = np.load('train_y.npy').reshape(-1,1)
test_x = np.load('test_x.npy').reshape(-1,900)
test_y = np.load('test_y.npy').reshape(-1,1)





class logistic:
	def __init__(self,theta,lr):
		self.theta = theta
		self.lr = lr
	def SGD(self,x,y,batch_size,steps):
		for step in range(steps):
			if(step % 100000 == 0):
				print(step," steps, SGD")
			batch_start = (batch_size*step)%(y.shape[0])
			batch_x = x[batch_start:batch_start+batch_size]
			batch_y = y[batch_start:batch_start+batch_size]
			g = -np.matmul(np.transpose(batch_x),batch_y)
			g /= 1+np.exp(np.matmul(np.transpose(np.matmul(batch_x,self.theta)),batch_y))
			self.theta = self.theta - self.lr*g
	def langevin(self,x,y,batch_size,steps):
		for step in range(steps):
			if(step % 100000 == 0):
				print(step," steps, langevin")
			epsilon = np.random.normal(0,0.01,(self.theta.shape[0],1))
			batch_start = (batch_size*step)%(y.shape[0])
			batch_x = x[batch_start:batch_start+batch_size]
			batch_y = y[batch_start:batch_start+batch_size]
			g = -np.matmul(np.transpose(batch_x),batch_y)
			g /= 1+np.exp(np.matmul(np.transpose(np.matmul(batch_x,self.theta)),batch_y))
			self.theta = self.theta - self.lr*g + np.sqrt(self.lr)*epsilon
	def predict(self,x):
		pre_y = 1/(1+np.exp(-np.matmul(x,self.theta)))
		pre_y = np.where(pre_y>=0.5,1,-1)
		pre_y = pre_y.astype('int32')
		return pre_y

randnum = random.randint(0,100)
random.seed(randnum)
random.shuffle(train_x)
random.seed(randnum)
random.shuffle(train_y)
theta=np.zeros((900,1))
m = logistic(theta,0.001)
m.SGD(train_x,train_y,1,2000000)
np.save("param_logistic.npy",m.theta)
m1 = logistic(theta,0.001)
m1.langevin(train_x,train_y,1,2000000)
from sklearn.metrics import accuracy_score
print("SGD acc:",accuracy_score(m.predict(test_x),test_y))
print("Langevin acc:",accuracy_score(m1.predict(test_x),test_y))
				



