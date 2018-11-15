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

def train(trainfile, devfile, dictionary, epochs=1, isAVG = 1):
    w_model = defaultdict(float)
    wa_model = defaultdict(float)
    features = set()

    train_set = list(tagger.readfile(trainfile))
    c = 1.

    for epoch in xrange(epochs):
        update = 0.
        error = 0.
        total_words = 0.
        for _, (wordseq, tagseq) in enumerate(train_set):
            total_words += len(wordseq)
            zseq = tagger.decode(wordseq, dictionary, w_model)
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
                        delta[t1, wordseq[i-1],wordseq[i+1]] += 1
                        delta[t2, wordseq[i-1],wordseq[i+1]] -= 1
                        features.add(t1, wordseq[i-1],wordseq[i+1])
                        features.add(t2, wordseq[i-1],wordseq[i+1])
                        error += 1
                    if t1 != t2 or tagseq[i-2] != zseq[i-2]:
                        delta[(tagseq[i-2],tagseq[i-1]), t1] += 1
                        delta[(zseq[i-2],zseq[i-1]), t2] -= 1
                        features.add(((tagseq[i-2],tagseq[i-1]), t1))
                        features.add(((zseq[i-2],zseq[i-1]), t2))
                #w_model += delta
                w_model = dictadd(w_model, delta)
                if isAVG:
                    wa_model = dictadd(wa_model, delta, c)
        #train_err = tagger.test(trainfile,dictionary, w_model)
        train_err = error/total_words
        dev_err = tagger.test(devfile, dictionary,w_model)
        feature_size = len(features)
        #print total_words
        if not isAVG:
            print "epoch %d, update %d,featrues %d, train_err %.2f%%, dev_err %.2f%%" % (epoch+1, update,feature_size, 100*train_err, 100*dev_err )
        else:
            #train_avg_err = tagger.test(trainfile,dictionary, dictadd(w_model, wa_model, -1/c))
            dev_AVG_err = tagger.test(devfile,dictionary, dictadd(w_model, wa_model, -1/c))
            print "epoch %d, update %d,featrues %d, train_err %.2f%%, dev_err %.2f%%, dev_avg %.2f%%" % (epoch+1, update, feature_size,100*train_err, 100*dev_err, 100*dev_AVG_err)
    if not isAVG:
        return w_model
    else:
        return dictadd(w_model, wa_model, -1/c)

# epoch 1 updates 102, |W|=291, train_err 3.90%, dev_err 9.36% avg_err 6.14%

if __name__ == "__main__":

    trainfile, devfile = sys.argv[1:3]
    dictionary, _ = tagger.mle(trainfile)
    t0 = time()
    model = train(trainfile, devfile, dictionary,20, 1)
    t1 = time()
    #print "time: %.2f s" % (t1 - t0)

    print "train_time {:.2f}".format(t1 - t0)
