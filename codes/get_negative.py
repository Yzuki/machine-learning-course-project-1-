import numpy as np
import math
import cv2
import os
os.mkdir('negative_train')
os.mkdir('negative_test')
def get_sample(x_list,y_list,img):
	sample = []
	for y in y_list:
		sample.append([])
	for y in range(len(y_list)):
		for x in range(len(x_list)):
			sample[y].append(img[y_list[y]][x_list[x]])
	sample = np.array(sample)
	res = cv2.resize(sample,(96,96))
	return res

def get_negative(x_list,y_list,img):
	negatives = []
	x_len = len(x_list)
	y_len = len(y_list)
	negatives.append(get_sample(x_list[:int(3/5*x_len)],y_list[:int(3/5*y_len)],img))
	negatives.append(get_sample(x_list[int(1/5*x_len):int(4/5*x_len)],y_list[:int(3/5*y_len)],img))
	negatives.append(get_sample(x_list[int(2/5*x_len):],y_list[:int(3/5*y_len)],img))
	negatives.append(get_sample(x_list[:int(3/5*x_len)],y_list[int(1/5*y_len):int(4/5*y_len)],img))
	negatives.append(get_sample(x_list[int(2/5*x_len):],y_list[int(1/5*y_len):int(4/5*y_len)],img))
	negatives.append(get_sample(x_list[:int(3/5*x_len)],y_list[int(2/5*y_len):],img))
	negatives.append(get_sample(x_list[int(1/5*x_len):int(4/5*x_len)],y_list[int(2/5*y_len):],img))
	negatives.append(get_sample(x_list[int(2/5*x_len):],y_list[int(2/5*y_len):],img))
	return negatives

for tmp1 in range(1,5):
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
			x_start = int(center_x - 5/3*minor_axis_radius)
			x_end = int(center_x + 5/3*minor_axis_radius)
			y_start = int(center_y - 5/3*major_axis_radius)
			y_end = int(center_y + 5/3*major_axis_radius)
			x_list = list(range(x_start,x_end))
			y_list = list(range(y_start,y_end))
			for x in range(len(x_list)):
				x_list[x] = max(x_list[x],0)
				x_list[x] = min(x_list[x],x_limit-1)
			for y in range(len(y_list)):
				y_list[y] = max(y_list[y],0)
				y_list[y] = min(y_list[y],y_limit-1)
			negatives = get_negative(x_list,y_list,img)
			for n in range(len(negatives)):
				cv2.imwrite('negative_train/'+str(i)+'_'+str(j)+'_'+str(n)+'.jpg', negatives[n])
		i+=facenum

for tmp1 in [10]:
	tmp2 = str(tmp1)
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
			x_start = int(center_x - 5/3*minor_axis_radius)
			x_end = int(center_x + 5/3*minor_axis_radius)
			y_start = int(center_y - 5/3*major_axis_radius)
			y_end = int(center_y + 5/3*major_axis_radius)
			x_list = list(range(x_start,x_end))
			y_list = list(range(y_start,y_end))
			for x in range(len(x_list)):
				x_list[x] = max(x_list[x],0)
				x_list[x] = min(x_list[x],x_limit-1)
			for y in range(len(y_list)):
				y_list[y] = max(y_list[y],0)
				y_list[y] = min(y_list[y],y_limit-1)
			negatives = get_negative(x_list,y_list,img)
			for n in range(len(negatives)):
				cv2.imwrite('negative_test/'+str(i)+'_'+str(j)+'_'+str(n)+'.jpg', negatives[n])
		i+=facenum