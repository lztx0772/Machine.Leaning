#!/usr/bin/env python
from __future__ import division
from collections import defaultdict
import sys
import tagger
from time import time
startsym, stopsym = "<s>", "</s>"

def dictadd(model, delta, c = 1):
    l_model = model.copy()
    for key, value in delta.iteritems(): 
        l_model[key] += value * c
    return l_model

def train3(trainfile, devfile, dictionary, epochs=1, isAVG = 1):
    w_model = defaultdict(float)
    wa_model = defaultdict(float)

    features = set()

    train_set = list(tagger.readfile(trainfile))
    c = 1.
    best_model = defaultdict(float)
    best_err = float("-inf")

    for epoch in xrange(epochs):
        update = 0.
        error = 0.
        total_words = 0.
        for _, (wordseq, tagseq) in enumerate(train_set):
            total_words += len(wordseq)
            zseq = tagger.decode3(wordseq, dictionary, w_model)
            if isAVG:
                c += 1.
            if zseq != tagseq:
                delta = defaultdict(float)
                wordseq = [startsym]*2 + wordseq + [stopsym]*2
                tagseq = [startsym]*2 + tagseq + [stopsym]*2
                zseq = [startsym]*2 + zseq + [stopsym]*2
                update += 1

                for i, (w, t1, t2) in enumerate(zip(wordseq,tagseq,zseq)[2:],2):
                    if t1 != t2:
                        delta[t1, w] += 1
                        delta[t2, w] -= 1
                        features.add((t1, w))
                        features.add((t2, w))
                        error += 1
                    if t1 != t2 or tagseq[i-2] != zseq[i-2] or tagseq[i-1 != zseq[i-1]]:
                        delta[(tagseq[i-2],tagseq[i-1]), t1] += 1
                        delta[(zseq[i-2],zseq[i-1]), t2] -= 1
                        features.add(((tagseq[i-2],tagseq[i-1]), t1))
                        features.add(((zseq[i-2],zseq[i-1]), t2))
                #w_model += delta
                #print delta
                #print features
                w_model = dictadd(w_model, delta)
                if isAVG:
                    wa_model = dictadd(wa_model, delta, c)
        #train_err = tagger.test(trainfile,dictionary, w_model)
        train_err = error/total_words
        dev_err = tagger.test3(devfile, dictionary,w_model)

        feature_size = len(features)
        #print total_words
        if not isAVG:
            print "epoch %d, update %d,featrues %d, train_err %.2f%%, dev_err %.2f%%" % (epoch+1, update,feature_size, 100*train_err, 100*dev_err )
            if dev_err < best_err:
                best_err = dev_err
                best_model = w_model.copy()
        else:
            #train_avg_err = tagger.test3(trainfile,dictionary, dictadd(w_model, wa_model, -1/c))
            dev_AVG_err = tagger.test3(devfile,dictionary, dictadd(w_model, wa_model, -1/c))
            if dev_AVG_err < best_err:
                best_err = dev_AVG_err
                best_model = dictadd(w_model, wa_model, -1/c)
            print "epoch %d, update %d,featrues %d, train_err %.2f%%, dev_err %.2f%%, dev_avg %.2f%%" % (epoch+1, update, feature_size,100*train_err, 100*dev_err, 100*dev_AVG_err)
    return best_model

def output_dev(inputfile, dictionary, model):
    dev_set = list(tagger.readfile(inputfile))
    #print dev_set
    f = open("dev.lower.unk.best", "w")
    for wordseq, tagseq in dev_set:
        best_tag = tagger.decode3(wordseq,dictionary, model)
        output = ""
        for word, tag in zip(wordseq, best_tag):
            output = output + word + "/"+ tag + " "
        print >> f, output
    f.close

def myreadfile(filename):
    for line in open(filename):
        wordtags = line.split()
        yield [w for w in wordtags]

def output_test(inputfile, dictionary, model):
    test_set = list(myreadfile(inputfile))
    #print dev_set
    f = open("test.lower.unk.best", "w")
    for wordseq in test_set:
        best_tag = tagger.decode3(wordseq,dictionary, model)
        output = ""
        for word, tag in zip(wordseq, best_tag):
            output = output + word + "/"+ tag + " "
        print >> f, output
    f.close

'''
def test3(filename, dictionary, model):    
    
    errors = tot = 0
    for words, tags in readfile(filename):
        mytags = decode3(words, dictionary ,model)
        #print mytags
        errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        tot += len(words) 
        
    return errors/tot
'''
if __name__ == "__main__":

    trainfile, devfile = sys.argv[1:3]
    dictionary, _ = tagger.mle(trainfile)
    t0 = time()
    model = train3(trainfile, devfile, dictionary,10, 1)
    t1 = time()
    #print "time: %.2f s" % (t1 - t0)

    testfile = "test.txt.lower.unk.unlabeled" 
   # output(testfile, devfile, dictionary, model)
    output_dev(devfile, dictionary, model)
    output_test(testfile, dictionary, model)

    print "train_time {:.2f}".format(t1 - t0)
