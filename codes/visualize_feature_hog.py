import numpy as np
import sklearn
from sklearn.manifold import TSNE
import os,gc
import cv2
import matplotlib.pyplot as plt
import matplotlib



X = np.load('test_x.npy')
labels = np.load('test_y.npy').reshape(-1,1)

tsne = TSNE(n_components=2,verbose = 1)
Y = tsne.fit_transform(X)

colors=['b', 'c']
idx_1 = [i1 for i1 in range(len(labels)) if labels[i1]==1]
flg1=plt.scatter(Y[idx_1,0], Y[idx_1,1], 20,color=colors[0],label='1')
idx_2= [i2 for i2 in range(len(labels)) if labels[i2]==-1]
flg2=plt.scatter(Y[idx_2,0], Y[idx_2,1], 20,color=colors[1], label='-1')

plt.legend()
plt.show()