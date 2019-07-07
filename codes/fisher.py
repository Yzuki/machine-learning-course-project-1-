import numpy as np
import math
import cv2
import random
import sklearn
train_x = np.load('train_x.npy').reshape(-1,900)
train_y = np.load('train_y.npy').reshape(-1,1)
test_x = np.load('test_x.npy').reshape(-1,900)
test_y = np.load('test_y.npy').reshape(-1,1)





class fisher:
        def __init__(self):
                self.w = None
                self.b = None
        def LDA(self,x,y):
                total = y.shape[0]
                pos = np.extract(y==1,y).shape[0]
                neg = total-pos
                x_pos = x[:pos]
                x_neg = x[pos:]
                m_pos = np.reshape(np.mean(x_pos,axis = 0),[1,900])
                m_neg = np.reshape(np.mean(x_neg,axis = 0),[1,900])
                
                S_w = np.dot(np.subtract(x_pos,m_pos).T,np.subtract(x_pos,m_pos)) + np.dot(np.subtract(x_neg,m_neg).T,np.subtract(x_neg,m_neg))
                inv = np.linalg.pinv(S_w)
                self.w = np.matmul(inv,(m_pos-m_neg).T)
                S_b = np.matmul((m_pos - m_neg).T,m_pos - m_neg)

                inter = np.matmul(np.matmul(self.w.T,S_b),self.w)
                intra = np.matmul(np.matmul(self.w.T,S_w),self.w)

                pos_mean = np.matmul(m_pos,self.w)
                neg_mean = np.matmul(m_neg,self.w)
                self.b = (pos_mean + neg_mean) / 2
                return inter,intra
                

        def predict(self,x):
                pre_y = np.matmul(x,self.w) - self.b
                pre_y = np.where(pre_y>=0,1,-1)
                pre_y = pre_y.astype('int32')
                return pre_y

m = fisher()
inter,intro = m.LDA(train_x,train_y)
print("Inter: ",inter, ", Intro: ",intro)
from sklearn.metrics import accuracy_score
print("fisher acc:",accuracy_score(m.predict(test_x),test_y))

