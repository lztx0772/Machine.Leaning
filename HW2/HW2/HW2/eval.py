from numpy import *
from decompose import *

def eval(filename="",basic=1000,epochs=5,MIRA=False,avg=False,outputfile="test.txt",p=0):
	X,Y,model = decompostion("income.train.txt.5k",[],False,False)
	X1,Y1,model1 = decompostion("income.dev.txt",model,False,False)
	
	file = open(outputfile,'w')
	# w is the model
	# output err_rate
	def check(w):
		result=[]
		for i in range(len(X1)):
			if dot(X1[i],w) * Y1[i] <=0:
				y=-1
			else:
				y=1
			result.append(y)
		return float(result.count(-1))/len(result)
	#default setting
	#N is the length of the training set
	#lr is the learinng rate
	#w is the original vector and w = 0 
	#D is the dimension of model
	result = dict()
	N = len(Y)
	D = len(X[0])
	w = zeros(D)
	_w = zeros(D)
	#
	_len = 0 
	for t in range(epochs):
		for i in range(N):
			_len = _len+1
			if dot(X[i],w) * Y[i] <= p:#negetive then update
				if MIRA:
					delta_w = (Y[i]-dot(w,X[i]))/dot(X[i],X[i])*X[i]
				else:
					delta_w = X[i] * Y[i] 
				w = w + delta_w
				_w = _w + _len * delta_w
			if (_len+1)%basic == 0:
				if avg:
					w_a = w - _w/(_len+1)
					err_rate = check(w_a)
				else:
					err_rate = check(w)
				pos = float(_len)/N
				result[pos] = err_rate
				print '{:.3f} epochs,error rate is {:.2%}'.format(float(_len)/N,err_rate)
				file.write('{:.3f}\t{:.2%}\n'.format(float(_len)/N,err_rate))

	min_p,min_v = min(result.items(), key=lambda x: x[1]) 
	print ' min at {:.3f} epochs,error rate is {:.2%}'.format(min_p,min_v)
	
	#print w,len(w)

if __name__ == "__main__":
	#def eval(filename="",basic=1000,epochs=5,MIRA=False,avg=False,outputfile="test.txt",p=0):
	#1.2
	#eval("",1000,5,False,False,"avg_train.txt",0)
	#1.4
	eval("",1000,5,False,True,"avg_train.txt",0)
	eval("",1000,5,True,True,"avg_train.txt",0.9)
	#1.6
	# eval("",200,1,False,True,"avg_train.txt",0)
	# eval("",200,1,False,False,"vanilla_train.txt",0)
	#2.1
	# eval("",1000,5,True,False,"MIRA",0)
	# eval("",1000,5,True,True,"avg_mira",0)
	#2.2
	# eval("",1000,5,True,False,"amira.1",0.1)
	# eval("",1000,5,True,False,"amira.5",0.5)
	# eval("",1000,5,True,False,"amira.9",0.9)
	# eval("",1000,5,True,True,"avg_amira.1",0.1)
	#eval("",1000,5,True,True,"avg_amira.5",0.5)
	# eval("",1000,5,True,True,"avg_amira.9",0.9)
	# 3(1)
	# eval("",1000,5,False,True,"reorder_per.txt",0)
	# eval("",1000,5,True,True,"reorder_mira.txt",0)
	# eval("",1000,5,False,True,"shuffle_per.txt",0)
	# eval("",1000,5,True,True,"shuffle_mira.txt",0)
	# #
	# eval("",1000,5,True,False,"reorder_mira.txt",0.1)
	# eval("",1000,5,True,False,"reorder_mira.txt",0.5)
	# eval("",1000,5,True,False,"reorder_mira.txt",0.9)
	# eval("",1000,5,True,True,"reorder_mira.txt",0.1)
	# eval("",1000,5,True,True,"reorder_mira.txt",0.5)
	# eval("",1000,5,True,True,"reorder_mira.txt",0.9)


