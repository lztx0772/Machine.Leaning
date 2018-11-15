from __future__ import division
import string
import math
import prework
from sklearn.svm import SVC
from numpy import *
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import GaussianNB
tokenize = lambda doc: doc.lower().split(" ")

def test(dev_X, dev_Y,w):
    result=[]
    for i in range(len(dev_X)):
        if dot(dev_X[i],w) * dev_Y[i] <=0:
            y=-1
        else:
            y=1
        result.append(y)
    return float(result.count(-1))/len(result)

def eval(train_X,train_Y,dev_X,dev_Y,epochs):
    
    size = len(train_Y)
    D = len(train_X[0])
    w = zeros(D)
    _w = zeros(D)
    best = zeros(D)
    best_err = 1.
    count = 0 
    for epoch in xrange(epochs):
        err = 0
        for index in xrange(size):
            if dot(train_X[index],w) * train_Y[index] <= 0:
                err +=1
                lr = 1
                w = w + lr * train_X[index] * train_Y[index]
                _w = _w + (index+1) * lr * train_X[index] * train_Y[index]
            count += 1
        _w = w - _w/count
        dev_err = test(dev_X,dev_Y,_w)
        train_err = test(train_X,train_Y,_w)
        print "epoch {0:d}, dev_err {1:0.2%}, train_err {2:0.4%}".format((epoch+1),dev_err,train_err)
        if best_err > dev_err:
            dev_err = best_err
            best = _w

    return _w 

def eval_svm_pegasos(train_X,train_Y,dev_X,dev_Y,epochs,C):
    
    size = len(train_Y)
    D = len(train_X[0])
    w = zeros(D)
    best = zeros(D)
    best_err = 1.
    count = 0 
    for epoch in xrange(epochs):
        for index in xrange(size):
            count+=1
            #print dot(train_X[index],w) * train_Y[index]
            if dot(train_X[index],w) * train_Y[index] < 1:
                nta = float(2) /(size*C)
                ita = float(1)/ (nta*count)
                delta_w = ita*(nta*w-train_Y[index]*train_X[index])
            else:
                delta_w = float(1)/count * w
            #print delta_w
            w = w - delta_w

        #print w
        dev_err = test(dev_X,dev_Y,w)
        train_err = test(train_X,train_Y,w)
        print "epoch {0:d}, dev_err {1:0.2%}, train_err {2:0.4%}".format((epoch+1),dev_err,train_err)
        if best_err > dev_err:
            best_err = dev_err
            best = w
    
    return w

if __name__ == "__main__":
    _,_, dev_mark, dev_documents = prework.getdata("reviews_Tools",30000,0.2)
    #train_mark, train_documents, dev_mark, dev_documents = prework.getdata("videoGame",20000,0.2)
    train_mark, train_documents, _,_ = prework.getdata("reviews_Tools",10000,0.2)#percentage of dev

    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, 
                                    tokenizer=tokenize,stop_words =['','r','l','a','the','and'],ngram_range=(1, 2))
    #print sklearn_tfidf.get_stop_words()
    sklearn_representation = sklearn_tfidf.fit_transform(train_documents)
    sklearn_representation_dev = sklearn_tfidf.transform(dev_documents)
    print float(sum(i==1 for i in dev_mark))/len(dev_mark)
    #print sklearn_tfidf.get_feature_names(),len(sklearn_tfidf.get_feature_names())#features
    #print sklearn_tfidf.get_stop_words()
    #print sklearn_representation.toarray() # X 
    # print len(sklearn_representation.toarray()[0]), len(sklearn_representation.toarray()[1])
    # print "2"
    # perceptron = smart_avg_perceptron(sklearn_representation.toarray(),train_mark);
    print "----average-perceptron--------"
    perceptron = eval(sklearn_representation.toarray(),train_mark,
                        sklearn_representation_dev.toarray(),dev_mark,
                        10)
    
    max_5 = perceptron.argsort()[-10:][::-1]
    for index in max_5:
        print sklearn_tfidf.get_feature_names()[index]
    min_5 = perceptron.argsort()[:10]
    for index in min_5:
        print sklearn_tfidf.get_feature_names()[index]
    C = 3
    print "----svm(pegasos) C={0:d}--------".format(C)
    perceptron = eval_svm_pegasos(sklearn_representation.toarray(),train_mark,
                                    sklearn_representation_dev.toarray(),dev_mark,
                                    10,C)
    max_5 = perceptron.argsort()[-10:][::-1]
    for index in max_5:
        print sklearn_tfidf.get_feature_names()[index]
    min_5 = perceptron.argsort()[:10]
    for index in min_5:
        print sklearn_tfidf.get_feature_names()[index]
    C = 2 
    print "----svm(pegasos) C={0:d}--------".format(C)
    perceptron = eval_svm_pegasos(sklearn_representation.toarray(),train_mark,
                                    sklearn_representation_dev.toarray(),dev_mark,
                                    10,C)
    max_5 = perceptron.argsort()[-10:][::-1]
    for index in max_5:
        print sklearn_tfidf.get_feature_names()[index]
    min_5 = perceptron.argsort()[:10]
    for index in min_5:
        print sklearn_tfidf.get_feature_names()[index]
    print "FINISH"
    # clf = GaussianNB()
    # clf.fit(sklearn_representation.toarray(),train_mark)
    # y = sum(x==y for x,y in zip(clf.predict(sklearn_representation_dev.toarray()),dev_mark))
    # print y
    # print float(y)/len(dev_mark)