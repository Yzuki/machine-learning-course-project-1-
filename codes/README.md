Unzip FDDB-folds.gz and originalPics.tar in the same directory as following codes.



Environments:

​	python 3.6.7, keras 2.2.4, sklearn 0.21.2



Preprocessing:(run in sequence)

​	get_positive.py: get positive samples

​	get_negative.py: get negative samples

​	get_hog.py: extract and visualize hog features

​	Dataset.py: merge positive and negative hog features to training set and testing set



Experiments:

​	visualize.py : visualize positive and negative samples

​	logistic.py: logistic model

​	fisher.py: fisher model

​	SVM.py: svm models and visualization of support vectors

​	CNN.py: CNN model

​	visualize_feature_hog.py: using t-SNE to visualize hog feature distributions

​	detection.py: face detection