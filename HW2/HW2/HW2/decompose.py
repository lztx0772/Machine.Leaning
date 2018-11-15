import sys
from numpy import *


def decompostion(file_name="income.train.txt.5k",model=[],data_reorder=False,shuffle=False):
	#get the file data 
	file = open(file_name,'r')
	list = file.readlines()
	file.close()
	#read and change

	if model == []:
		set_age = set()
		set_work = set()
		set_degree = set()
		set_marriage = set()
		set_experience = set()
		set_race = set()
		set_gender = set()
		set_hours = set()
		set_state = set(["Holand-Netherlands","Hungary"])
		set_earn = set()
		#read the data 
		#this part would be useless become the model could be design by people not just meachine 
		bias = 0 
		_bias = []
		for line in list:
			#read
			age,work_for,degree,marriage,experience,race,gender,hours,state,earn = line.strip().split(', ')
			set_age.add(age)
			set_work.add(work_for)
			set_degree.add(degree)
			set_marriage.add(marriage)
			set_experience.add(experience)
			set_race.add(race)
			set_gender.add(gender)
			set_hours.add(hours)
			set_state.add(state)
			set_earn.add(earn)
			_bias.append(earn)
		
		b = _bias.count(">50K")-_bias.count("<=50K")
		set_age = range(1,101)
		set_work = sorted(set_work)
		set_degree = sorted(set_degree)
		set_marriage = sorted(set_marriage)
		set_experience = sorted(set_experience)
		set_race = sorted(set_race)
		set_gender = sorted(set_gender)
		set_hours = range(1,101)
		set_state = sorted(set_state)

		model.extend(set_age)
		model.extend(set_work)
		model.extend(set_degree)
		model.extend(set_marriage)
		model.extend(set_experience)
		model.extend(set_race)
		model.extend(set_gender)
		model.extend(set_hours)
		model.extend(set_state)
		#model.append("bias")#bias

	#D = len(set_age)+ len(set_work) + len(set_degree) + len(set_marriage) + len(set_experience) + len(set_race) + 1 + len(set_hours) + len(set_state)
	#
	# age   | work       | degree   | marriage  | experience | race    | gender    | hours   | state    |  bias
	# 0-99  | 100-106    | 107-122  | 123-129   | 130-143    | 144-148 | 149,150   | 151-250 | 251- 291 |  292
	# 
	#
	
	X=[]
	Y=[]
	D=len(model)
	#c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 'Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay', '10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college', 'Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White', 'Female', 'Male', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Honduras', 'Hong', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia', 'bias']
	#for i in range(D):
	#	print model[i],c[i]

	for line in list:
	 	x = array([-1]*D)
	 	age,work_for,degree,marriage,experience,race,gender,hours,state,earn = line.strip().split(', ')
	 	x[int(age)-1]=1
	 	x[model.index(work_for)]=1
	 	x[model.index(degree)]=1
	 	x[model.index(marriage)]=1
	 	x[model.index(experience)]=1
	 	x[model.index(race)]=1
	 	x[model.index(gender)]=1
	 	x[int(hours)+150]=1
	 	x[model.index(state)]=1
	 	#x[292]=1
	 	if data_reorder:
	 		if earn == '>50K':
	 			X = [x]+X
	 			Y = [1]+Y
	 		else:
	 			X.append(x)
	 			Y.append(-1)
	 	else:
		 	X.append(x)
		 	y = -1
		 	if earn == '>50K':
		 		y = 1
		 	Y.append(y)
	#print float(Y.count(-1))/len(Y)
	if shuffle:
		N = range(len(X))
		random.shuffle(N)
		X1 = []
		Y1 = []
		for index in N:
			X1.append(X[index])
			Y1.append(Y[index])

		return X1,Y1,model

	return X,Y,model
		#print strings
if __name__ == "__main__":
	X,Y,model = decompostion()
	print float(Y.count(1))/len(Y)