import sys
from collections import defaultdict
import numpy as np
from matplotlib import pyplot
from sklearn import svm
import time

def map_data(filename, feature2index):
    data = []
    dimention = len(feature2index)
    for j, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(len(feature2index))-1
        # print feat_vec
        for i, fv in enumerate(features):
            if (i, fv) in feature2index:
                feat_vec[feature2index[i, fv]] = 1
        feat_vec[0] = 1

        data.append((feat_vec, 1 if features[-1] == ">50K" else -1))
    return data


def create_feature_map(train_file):
    column_values = defaultdict(set)
    for line in open(train_file):
        line = line.strip()
        features = line.split(", ")[:-1]
        for i, fv in enumerate(features):
            column_values[i].add(fv)
    feature2index = {(-1, 0): 0}
    for i, value in column_values.iteritems():
        for v in value:
            feature2index[i, v] = len(feature2index)
    return feature2index

def decompose(datas):
    data_X=[]
    data_Y=[]
    for i,j in datas:
        data_X.append(i)
        data_Y.append(j)
    return np.array(data_X),np.array(data_Y)

def run_svm(train_datas,c=1):
    train_X,train_Y=decompose(train_datas)
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(train_X, train_Y)
    return clf

def eval(X,Y,clf):
    return 1-clf.score(X,Y)

def non_zero_slacks(clf,c):
    count=0
    for i in clf.dual_coef_[0]:
        if abs(i-C)<=1e-6:
            count+=1
    return count

def count_slacks(X,Y,clf,C):
    sum_slacks = sum(1-clf.decision_function(X))
    return sum_slacks,np.dot(clf.coef_[0],clf.coef_[0])*0.5 + C*sum_slacks

def five_pos(X,Y,clf):
    list = 1 - clf.decision_function(X)
    pos_list = []
    neg_list = []
    for i,slacks in enumerate(list):
        if Y[i][1] >0:
            pos_list.append((Y[i][0],slacks))
        elif Y[i][1]<0:
            neg_list.append((Y[i][0],slacks))
    pos_list=sorted(pos_list,key=lambda x: x[1],reverse=True)[:5]
    neg_list=sorted(neg_list,key=lambda x: x[1],reverse=True)[:5]
    return pos_list,neg_list


if __name__ == "__main__":
    feature2index = create_feature_map("income.train.txt")
    train_datas = map_data("income.train.txt.5k",feature2index)
    dev_datas = map_data("income.dev.txt",feature2index)
    train_X, train_Y = decompose(train_datas)
    dev_X, dev_Y = decompose(dev_datas)

    C=1
    clf = run_svm(train_datas,C)


    # 1.1
    print
    print "train error rate:{:.2%}".format(eval(train_X,train_Y,clf))
    # 1.2
    print
    print "the number of support vectors:{}".format(sum(clf.n_support_)), "the non_zero slacks are {}".format(non_zero_slacks(clf,C))
    # 1.3
    print
    sum_slacks, total=count_slacks(clf.support_vectors_,train_Y,clf,C)
    print "the total amount violations is:{}, objective:{}".format(sum_slacks,total)
    # 1.4
    print
    sv_Y = []
    for i,j in enumerate(clf.support_):
        sv_Y.append((j,train_Y[j]))
    pos_list,neg_list = five_pos(clf.support_vectors_,sv_Y,clf)
    lines = open("income.train.txt.5k").readlines()
    print "5 most positive:"
    for i,j in pos_list:
        print lines[i][:-1]," slack: {}".format(j)
    print "5 most negtive:"
    for i, j in neg_list:
        print lines[i][:-1], " slack: {}".format(j)
    #1.5
    print "Vary C:"
    list_c = [0.01,0.1,1,2,5,10]
    for i in list_c:
        vary_clf = svm.SVC(kernel='linear', C=i)
        time_s = time.time()
        vary_clf.fit(train_X,train_Y)
        time_end = time.time()
        print "C = {},training time: {}, training error: {:.2%}, dev error: {:.2%}, number of support vectors:{}"\
                .format(i,\
                time_end-time_s,\
                1-vary_clf.score(train_X,train_Y),\
                1-vary_clf.score(dev_X,dev_Y),\
                sum(vary_clf.n_support_))

    #3.1
    print
    qua_clf = svm.SVC(kernel='poly', degree=2,coef0=1,C=1)
    time_s = time.time()
    qua_clf.fit(train_X,train_Y)
    time_end = time.time()
    print "C = 1,training time: {}, training error: {:.2%}, dev error: {:.2%}"\
        .format(time_end-time_s,\
                1-qua_clf.score(train_X,train_Y),\
                1-qua_clf.score(dev_X,dev_Y))
    #3.3
    print
    list_c = [0.01,0.1,1,2,5,10]
    for i in list_c:
        qua_clf = svm.SVC(kernel='poly', degree=2, coef0=1, C=i)
        time_s = time.time()
        qua_clf.fit(train_X,train_Y)
        time_end = time.time()
        print "C = {},training time: {}, training error: {:.2%}, dev error: {:.2%}, number of support vectors:{}"\
            .format(i,\
                    time_end - time_s, \
                    1 - qua_clf.score(train_X, train_Y), \
                    1 - qua_clf.score(dev_X, dev_Y),\
                    sum(qua_clf.n_support_))