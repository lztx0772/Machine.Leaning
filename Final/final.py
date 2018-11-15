#!/usr/bin/env python

from __future__ import division
from collections import defaultdict
import sys
from math import log

def readfile(filename):
    flag = 0
    index, rates, comments = []

    text = open(filename).readlines()[:10000]
    for line in text:
        if flag % 3 == 0 :
            index += [line]
            flag += 1
        elif flag % 3 == 1 :
            flag += 1
            rates += [line]
        else:
             flag += 1
             comments += [line]
        #print line
    return rates, comments
    #return 1, 2
            
if __name__ == "__main__":
    inputfile = "videoGame"
    index, rates, comments = readfile(inputfile)
    print rates[2]
    print comments[2]