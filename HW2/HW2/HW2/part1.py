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
		if dot(w,tmp) * Y[i] <=0:
			y=-1
		else:
			y=1
		result.append(y)
	return float(result.count(-1))/len(result)

def linear_svm(C=1):
	#get the file data 
	#file = open("income.train.txt.5k",'r')
	#examples = file.readlines()
	#file.close()
	train_X,train_Y,model = decompostion("income.train.txt.5k",[],False,False)
	#dev_X,dev_Y,_ = decompostion("income.dev.txt",model,False,False)

	t0 = time()

	clf = svm.SVC(kernel='linear', C=C)
	clf.fit(train_X,train_Y)
	print "support vectors"
	#print clf.support_vectors_
	
	svmmodel = np.concatenate((clf.coef_[0], clf.intercept_))#svm w
	#svmmodel = array(clf.coef_[0])
	# print "model (primal)"
	# print svmmodel
	# print clf.dual_coef_
	
	t1 = time()
	train_time = t1 - t0

	#train_err = check(svmmodel,train_X,train_Y)
	#dev_err	  = check(svmmodel,dev_X,dev_Y)
	# q1
	#print "train_err {:.2%}".format(train_err)
	#print "dev_err {:.2%}".format(dev_err)
	print "train_time" + " " + str(train_time)
	# q2
	#print clf.n_support_#,len(clf.support_vectors_),len(clf.support_vectors_[0]),len(clf.support_)
	count = 0
	#print len(clf.support_vectors_[0])
	for x in clf.support_vectors_:
		tmp = np.append(x,[1])
		if abs(dot(tmp,svmmodel))-1<=1e-6:
			count+=1
	#print count
	count= 0
	for ps, alpha in zip(clf.support_vectors_, clf.dual_coef_[0]):
		if abs(alpha) == C: # violating support vectors: alpha == C
			count+=1
		#else: # good support vectors: 0 < alpha < C
	#print count
	# q3
	primal_keci =0
	for i in range(len(train_X)):
		tmp = np.append(train_X[i],[1])
		delta = 1-dot(tmp,svmmodel)*train_Y[i]
		if delta<0:
			delta = 0
		primal_keci += delta
	total_violation = 0.5*dot(svmmodel,svmmodel)+primal_keci
	#print "total slacks " + str(primal_keci)
	#print "total violating " + str(total_violation)
	#q4 list 
	pos=[]
	neg=[]
	#print len(clf.support_)
	for index in xrange(len(clf.support_)):
		sign = train_Y[index]
		tmp = np.append(train_X[index],[1])
		slacks = 1 - dot(svmmodel,tmp)*sign
		if sign == 1:
			pos.append((slacks,index))
		else:
			neg.append((slacks,index))
	#print len(pos),len(neg)
	pos.sort()
	neg.sort()
	#print pos
'''
	print "pos 5"
	for _v,vectors in pos[-5:][::-1]:
		print examples[vectors],_v
	print "neg 5"
	for _v,vectors in neg[-5:][::-1]:
		print examples[vectors],_v
'''

if __name__ == '__main__':
	linear_svm()

