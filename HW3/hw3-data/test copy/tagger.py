#!/usr/bin/env python

from __future__ import division
from collections import defaultdict
import sys
from math import log
startsym, stopsym = "<s>", "</s>"

def readfile(filename):#read then split word/tag form to word and tag
    for line in open(filename):
        wordtags = map(lambda x: x.rsplit("/", 1), line.split())
        yield [w for w,t in wordtags], [t for w,t in wordtags] # (word_seq, tag_seq) pair
    
def mle(filename): # Max Likelihood Estimation of HMM
    twfreq = defaultdict(lambda : defaultdict(int)) # twfreq[][] tag to word
    ttfreq = defaultdict(lambda : defaultdict(int))  # tag to tag
    tagfreq = defaultdict(int)    
    dictionary = defaultdict(set)

    for words, tags in readfile(filename):#read file, add to tag-word tag-tag dict
        last = startsym
        tagfreq[last] += 1 #tag frequency
        for word, tag in zip(words, tags) + [(stopsym, stopsym)]:
            #if tag == "VBP": tag = "VB" # +1 smoothing
            twfreq[tag][word] += 1            
            ttfreq[last][tag] += 1
            dictionary[word].add(tag)
            tagfreq[tag] += 1
            last = tag            
    
    model = defaultdict(float)
    num_tags = len(tagfreq)
    for tag, freq in tagfreq.iteritems(): 
        logfreq = log(freq)
        for word, f in twfreq[tag].iteritems():
            model[tag, word] = log(f) - logfreq 
        logfreq2 = log(freq + num_tags)
        for t in tagfreq: # all tags
            model[tag, t] = log(ttfreq[tag][t] + 1) - logfreq2 # +1 smoothing
        
    return dictionary, model

def decode(words, dictionary, model):

    def backtrack(i, tag):
        if i == 1:
            return []
        return backtrack(i-1, back[i][tag]) + [tag[1]]

    words = [startsym]*2 + words + [stopsym]*2

    best = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    #best[0][(startsym,startsym)] = best[1][(startsym,startsym)] = 1
    best[1][(startsym,startsym)] = 1.
    back = defaultdict(dict)

    #print " ".join("%s/%s" % wordtag for wordtag in zip(words,tags)[1:-1])
    for i, word in enumerate(words[2:], 2):#1 is option
        #print i, " ", word
        for tag in dictionary[word]: #tag to words 
            for prev in best[i-1]: 
                    score = best[i-1][prev] + model[prev, tag] + model[tag, word]
                    if score > best[i][(prev[1],tag)]:
                        best[i][(prev[1],tag)] = score
                        back[i][(prev[1],tag)] = prev # prev = (prevprev, prev)
        #print i, word, dictionary[word], best[i]
    #print best[len(words)-1][stopsym]
    mytags = backtrack(len(words)-1, (stopsym,stopsym))[:-2]
    #print " ".join("%s/%s" % wordtag for wordtag in mywordtags)
    #print mytags
    #print best

    return mytags

def test(filename, dictionary, model):    
    
    errors = tot = 0
    for words, tags in readfile(filename):
        mytags = decode(words, dictionary ,model)
        #print mytags
        errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        tot += len(words) 
        
    return errors/tot
        
if __name__ == "__main__":
    trainfile, devfile = sys.argv[1:3]
    
    dictionary, model = mle(trainfile)

    print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
    print "dev_err {0:.2%}".format(test(devfile, dictionary, model))
