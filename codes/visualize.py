import numpy as np
import math
import cv2
import random
def get_sample(x_list,y_list,img):
	cv2.rectangle(img, (x_list[0],y_list[0]), (x_list[-1],y_list[-1]), (random.randint(0,255),random.randint(0,255),random.randint(0,255)), random.randint(0,10))

def get_negative(x_list,y_list,img):
	negatives = []
	x_len = len(x_list)
	y_len = len(y_list)
	get_sample(x_list[:int(3/5*x_len)],y_list[:int(3/5*y_len)],img)
	get_sample(x_list[int(1/5*x_len):int(4/5*x_len)],y_list[:int(3/5*y_len)],img)
	get_sample(x_list[int(2/5*x_len):],y_list[:int(3/5*y_len)],img)
	get_sample(x_list[:int(3/5*x_len)],y_list[int(1/5*y_len):int(4/5*y_len)],img)
	get_sample(x_list[int(2/5*x_len):],y_list[int(1/5*y_len):int(4/5*y_len)],img)
	get_sample(x_list[:int(3/5*x_len)],y_list[int(2/5*y_len):],img)
	get_sample(x_list[int(1/5*x_len):int(4/5*x_len)],y_list[int(2/5*y_len):],img)
	get_sample(x_list[int(2/5*x_len):],y_list[int(2/5*y_len):],img)

for tmp1 in [1]:
	tmp2 = str(tmp1)
	tmp2 = '0'+tmp2
	filename = 'FDDB-folds/FDDB-fold-'+tmp2+'-ellipseList.txt'
	f = open(filename)
	lines = f.readlines()
	i = 0
	imgpath = lines[i].strip()
	i+=1
	img = cv2.imread(imgpath+'.jpg')
	print(imgpath)
	img_positive = img.copy()
	x_limit = img.shape[1]
	y_limit = img.shape[0]
	facenum = 1
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
		get_negative(x_list,y_list,img)

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
		get_sample(x_list,y_list,img_positive)

		cv2.imshow('img_negative',img)
		cv2.imshow('img_positive',img_positive)
		cv2.waitKey(0)
		cv2.destroyAllWindows()