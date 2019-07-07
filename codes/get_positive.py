import numpy as np
import math
import cv2
import os
os.mkdir('positive_train')
os.mkdir('positive_test')
for tmp1 in range(1,9):
	tmp2 = str(tmp1)
	tmp2 = '0'+tmp2
	filename = 'FDDB-folds/FDDB-fold-'+tmp2+'-ellipseList.txt'
	f = open(filename)
	lines = f.readlines()
	i = 0
	while(i<len(lines)):
		imgpath = lines[i].strip()
		i+=1
		img = cv2.imread(imgpath+'.jpg')
		x_limit = img.shape[1]
		y_limit = img.shape[0]
		facenum = int(lines[i])
		i+=1
		for j in range(facenum):
			tmp = lines[i+j].split()
			major_axis_radius = float(tmp[0])
			minor_axis_radius = float(tmp[1])
			angle = tmp[2]
			center_x = float(tmp[3])
			center_y = float(tmp[4])
			x_start = int(center_x - 4/3*minor_axis_radius)
			x_end = int(center_x + 4/3*minor_axis_radius)
			y_start = int(center_y - 4/3*major_axis_radius)
			y_end = int(center_y + 4/3*major_axis_radius)
			x_list = list(range(x_start,x_end))
			y_list = list(range(y_start,y_end))
			for x in range(len(x_list)):
				x_list[x] = max(x_list[x],0)
				x_list[x] = min(x_list[x],x_limit-1)
			for y in range(len(y_list)):
				y_list[y] = max(y_list[y],0)
				y_list[y] = min(y_list[y],y_limit-1)
			positive_sample = []
			for y in y_list:
				positive_sample.append([])
			for y in range(len(y_list)):
				for x in range(len(x_list)):
					positive_sample[y].append(img[y_list[y]][x_list[x]])
			positive_sample = np.array(positive_sample)
			res = cv2.resize(positive_sample,(96,96))
			cv2.imwrite('positive_train/'+str(i)+'_'+str(j)+'.jpg', res)
		i+=facenum

for tmp1 in range(9,11):
	tmp2 = str(tmp1)
	if(tmp1==9):
		tmp2 = '0'+tmp2
	filename = 'FDDB-folds/FDDB-fold-'+tmp2+'-ellipseList.txt'
	f = open(filename)
	lines = f.readlines()
	i = 0
	while(i<len(lines)):
		imgpath = lines[i].strip()
		i+=1
		img = cv2.imread(imgpath+'.jpg')
		x_limit = img.shape[1]
		y_limit = img.shape[0]
		facenum = int(lines[i])
		i+=1
		for j in range(facenum):
			tmp = lines[i+j].split()
			major_axis_radius = float(tmp[0])
			minor_axis_radius = float(tmp[1])
			angle = tmp[2]
			center_x = float(tmp[3])
			center_y = float(tmp[4])
			x_start = int(center_x - 4/3*minor_axis_radius)
			x_end = int(center_x + 4/3*minor_axis_radius)
			y_start = int(center_y - 4/3*major_axis_radius)
			y_end = int(center_y + 4/3*major_axis_radius)
			x_list = list(range(x_start,x_end))
			y_list = list(range(y_start,y_end))
			for x in range(len(x_list)):
				x_list[x] = max(x_list[x],0)
				x_list[x] = min(x_list[x],x_limit-1)
			for y in range(len(y_list)):
				y_list[y] = max(y_list[y],0)
				y_list[y] = min(y_list[y],y_limit-1)
			positive_sample = []
			for y in y_list:
				positive_sample.append([])
			for y in range(len(y_list)):
				for x in range(len(x_list)):
					positive_sample[y].append(img[y_list[y]][x_list[x]])
			positive_sample = np.array(positive_sample)
			res = cv2.resize(positive_sample,(96,96))
			cv2.imwrite('positive_test/'+str(i)+'_'+str(j)+'.jpg', res)
		i+=facenum