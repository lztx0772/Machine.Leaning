#!/usr/bin/env python

import sys
import os
import math
import numpy as np
from random import random, randint
from matplotlib import pyplot
from sklearn import svm
from time import time

from decompose	import *
def check(w,X,Y):
	#print len(w),len(X[0])
	result=[]
	for i in range(len(X)):
		tmp = np.append(X[i],[1])
		#print tmp,len(tmp)
		if dot(w,tmp) <=0:
			y=-1
		else:
			y=1
		result.append(y)
	return float(result.count(1))/len(result)

def linear_svm(C=1):

	train_X,train_Y,model = decompostion("income.train.txt.5k",[],False,False)
	dev_X,dev_Y,_ = decompostion("income.dev.txt",model,False,False)
	#test_X,test_Y,_ = decompostion("")

	t0 = time()

	clf = svm.SVC(kernel='linear', C=C)
	clf.fit(train_X,train_Y)
	print "support vectors"
	#print clf.support_vectors_
	
	svmmodel = np.concatenate((clf.coef_[0], clf.intercept_))#svm w

	t1 = time()
	train_time = t1 - t0

	train_err = check(svmmodel,train_X,train_Y)
	dev_err	  = check(svmmodel,dev_X,dev_Y)

	print "train_time" + " " + str(train_time)
	print "train_pos" + " " + str(train_err)
	print "dev_pos" + " " + str(dev_err)



if __name__ == '__main__':
	linear_svm()

