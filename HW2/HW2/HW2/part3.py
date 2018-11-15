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
def check(Y, P_Y):
	result=[]
	for i in range(len(Y)):
		if Y[i]*P_Y[i] <=0:
			y=-1
		else:
			y=1
		result.append(y)
	return float(result.count(-1))/len(result)

def linear_svm(C=1):

	train_X,train_Y,model = decompostion("income.train.txt.5k",[],False,False)
	dev_X,dev_Y,_ = decompostion("income.dev.txt",model,False,False)

	t0 = time()

	clf = svm.SVC(kernel='poly', degree=2, coef0=1, C=C)
	clf.fit(train_X,train_Y)
	print "support vectors\t"+ str(len(clf.support_vectors_))
	#print len(clf.support_vectors_)

	t1 = time()
	train_time = t1 - t0

	predict_Y = clf.predict(train_X)
	predict_dev_Y = clf.predict(dev_X)

	train_err = check(train_Y,predict_Y)
	dev_err	  = check(dev_Y, predict_dev_Y)

	print "train_time {:.2f}".format(train_time)

	print "train_err {:.2%}".format(train_err)
	print "dev_err {:.2%}".format(dev_err)

if __name__ == '__main__':
	for c in [0.01, 0.1, 1, 2, 5, 10, 100, 200, 300, 400, 500, 600, 700, 800]:
		print "C :"  + str(c)
		linear_svm(c)

