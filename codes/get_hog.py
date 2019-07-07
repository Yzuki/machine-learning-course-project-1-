import os
import cv2
import numpy as np
from skimage import feature
folders = ['positive_train','negative_train','negative_test','positive_test']
img = None
hog = None
path = None
names = {}
for folder in folders:
	imgnames = os.listdir(folder)
	outs = []
	out_path = folder+'_hog'
	names[folder] = []
	for imgname in imgnames:
		names[folder].append(imgname)
		path = os.path.join(folder,imgname)
		img = cv2.imread(path)
		out,hog = feature.hog(
			img,orientations=9,
			pixels_per_cell = (16,16),
			cells_per_block = (2,2),
			visualize = True,
			feature_vector = True,
			multichannel = True)
		outs.append(out)
	outs = np.array(outs)
	np.save(out_path+'.npy',outs)

train_names = names['positive_train'] + names['negative_train']
test_names = names['positive_test'] + names['negative_test']
ff = open('train_names.txt','w')
for name in train_names:
	ff.write(name+'\n')
ff.close()
ff = open('test_names.txt','w')
for name in test_names:
	ff.write(name+'\n')
ff.close()
import matplotlib.pyplot as plt
from skimage import data, exposure

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(img)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()



