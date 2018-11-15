import re
import random
#in this file, it would do the preword, about remove punctuation, get data and split it into train and dev

def clear_punctuation(instring):
    import string
    translation_table = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    out = instring.translate(translation_table)
    #out = instring.translate(None,string.punctuation)
    
    return out

def remove_stop_word(instring):
    import re
    out = re.sub(r'\d+[A-Za-z0-9]*',"",instring)
    return out

def getdata(filename,amount,percent):
    file = open(filename,"r")
    lines = file.readlines()
    length = len(lines)/3
    train_review = []
    train_mark = []
    dev_review = []
    dev_mark = []
    count = 0 
    #set the dev set start point and end point
    #random_point = random.random()*(1-percent)
    random_point = 1*(1-percent)
    start = int(random_point*amount)
    end = start + int(percent*amount)
    #split the review mark to different type (basic two: positive and negative )
    File = open("tmp","w")
    for index in xrange(length):
        if index == amount:
            break
        sign = lines[3*index+1][:-1]
        if sign == "1.0" or sign == '5.0' or sign=="2.0":
            if sign=='1.0' or sign=="2.0": 
                _mark = -1
            else:
                _mark = 1
        File.write(lines[3*index+2][:-1]+'\n')
        _review = clear_punctuation(lines[3*index+2][:-1])
        _review = remove_stop_word(_review)
        File.write(_review+'\n')
        if index in xrange(start,end):
            dev_mark += [_mark]
            dev_review += [_review]
        else:
            train_mark += [_mark]
            train_review += [_review]
    
    return train_mark,train_review,dev_mark,dev_review
    
if __name__ == "__main__":
    getdata("videoGame")
