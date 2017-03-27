#!/usr/bin/env python
# encoding: utf-8

"""
@brief Phrase finding with spark
@param fg_year The year taken as foreground
@param f_unigrams The file containing unigrams
@param f_bigrams The file containing bigrams
@param f_stopwords The file containing stop words
@param w_info Weight of informativeness
@param w_phrase Weight of phraseness
@param n_workers Number of workers
@param n_outputs Number of top bigrams in the output
"""

import sys
from pyspark import SparkConf, SparkContext
from functools import partial
from math import log

def tokenize(fg_year,lines):
    tokens=lines.split("\t")
    if int(tokens[1])==fg_year:
        value=(int(tokens[2]), 0)
    else:
        value=(0, int(tokens[2]))
    return tokens[0], value


def delstop_words(stopwords, lines):
    words=lines[0].split()
    if (len(words)==1):
        if (words[0] not in stopwords):
            return lines
    else:
        if ((words[0] not in stopwords) and (words[1] not in stopwords)):
            return lines

def firstKeyReformat(lines):
    words=lines[0].split()
    key=words[0]
    value=(words[1], lines[1])
    return key, value

def groupedReformat1(lines):
    return  lines[1][0][0], (lines[0],lines[1][0][1][0], lines[1][0][1][1], lines[1][1][0])

def groupedReformat2(lines):
    return lines[1][0][0], lines[0], lines[1][0][1], lines[1][0][2],lines[1][0][3], lines[1][1][0]

def calculateP(B, U, fg_total_big, bg_total_big, fg_total_unig, lines):
    w1=lines[0] 
    w2=lines[1]
    fg_big=lines[2]
    bg_big=lines[3]
    fg_w1=lines[4]
    fg_w2=lines[5]
    pfg=(fg_big+1)*1.0/(B+fg_total_big)
    pbg=(bg_big+1)*1.0/(B+bg_total_big)
    fg1=(fg_w1+1)*1.0/(U+fg_total_unig)
    fg2=(fg_w2+1)*1.0/(U+fg_total_unig)

    key=w1+"-"+w2
    return key, pfg, pbg, fg1, fg2

def calculateInfoPhrase(w_info, w_phrase, lines):
    pfg=lines[1]
    pbg=lines[2]
    fg1=lines[3]
    fg2=lines[4]

    phrase=kldiv(pfg, fg1*fg2)
    info=kldiv(pfg, pbg)
    score=w_info*info+w_phrase*phrase
    return lines[0], score


def kldiv(p,q):
    r=p*(log(p)-log(q))
    return r

def getkey(item):
    return item[1]

def increment_counter():
    global counter
    counter += 1

def get_number_of_element(rdd):
    global counter
    counter = 0
    rdd.foreach(lambda x:increment_counter())
    return counter

def main(argv):
    # parse args
    fg_year = int(argv[1])
    f_unigrams = argv[2]
    f_bigrams = argv[3]
    f_stopwords = argv[4]
    w_info = float(argv[5])
    w_phrase = float(argv[6])
    n_workers = int(argv[7])
    n_outputs = int(argv[8])

    stop_words=["a", "about", "above", "across", "after","afterwards","again","against","all","almost","alone","along","already",\
                "also","although","always","am","among","amongst","amoungst","amount","an","and","another","any","anyhow",\
                "anyone","anything","anyway","anywhere","are","around","as","at","back","be","became","because","become",\
                "becomes","becoming","been","before","beforehand","behind","being","below","beside","besides","between","beyond",\
                "bill","both","bottom","but","by","call","can","cannot","cant","co","computer","con","could","couldnt","cry","de",\
                "describe","detail","do","done","down","due","during","each","eg","eight","either","eleven","else","elsewhere","empty",\
                "enough","etc","even","ever","every","everyone","everything","everywhere","except","few","fifteen","fify","fill","find",\
                "fire","first","five","for","former","formerly","forty","found","four","from","front","full","further","get","give",\
                "go","had","has","hasnt","have","he","hence","her","here","hereafter","hereby","herein","hereupon","hers","herself","him",\
                "himself","his","how","however","hundred","i","ie","if","in","inc","indeed","interest","into","is","it","its","itself","keep",\
                "last","latter","latterly","least","less","ltd","made","many","may","me","meanwhile","might","mill","mine","more","moreover",\
                "most","mostly","move","much","must","my","myself","name","namely","neither","never","nevertheless","next","nine","no","nobody",\
                "none","noone","nor","not","nothing","now","nowhere","of","off","often","on","once","one","only","onto","or","other","others",\
                "otherwise","our","ours","ourselves","out","over","own","part","per","perhaps","please","put","rather","re","same","see","seem",\
                "seemed","seeming","seems","serious","several","she","should","show","side","since","sincere","six","sixty","so","some","somehow",\
                "someone","something","sometime","sometimes","somewhere","still","such","system","take","ten","than","that","the","their","them",\
                "themselves","then","thence","there","thereafter","thereby","therefore","therein","thereupon","these","they","thick","thin","third",\
                "this","those","though","three","through","throughout","thru","thus","to","together","too","top","toward","towards","twelve",\
                "twenty","two","un","under","until","up","upon","us","very","via","was","we","well","were","what","whatever","when","whence","whenever",\
                "where","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","whoever",\
                "whole","whom","whose","why","will","with","within","without","would","yet","you","your","yours","yourself","yourselves"]

    """ configure pyspark """
    conf = SparkConf().setMaster('local[{}]'.format(n_workers))  \
                      .setAppName(argv[0])
    sc = SparkContext(conf=conf)


    bigramsRaw=sc.textFile(f_bigrams)
    big1=bigramsRaw.map(partial(tokenize, fg_year))
    big2=big1.filter(partial(delstop_words, stop_words))
    big3=big2.reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))

    unigramsRaw=sc.textFile(f_unigrams)
    unig1=unigramsRaw.map(partial(tokenize, fg_year))
    unig2=unig1.filter(partial(delstop_words, stop_words))

    unig3=unig2.reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))

    B=len(big3.map(lambda (x,y): x).distinct().collect())
    U=len(unig3.map(lambda (x,y):x).distinct().collect())
    

    (fg_total_big, bg_total_big)=big3.map(lambda (x,y):y).reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
    #print fg_total_big, bg_total_big
    (fg_total_unig, bg_total_unig)=unig3.map(lambda (x,y):y).reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
    #print fg_total_unig, bg_total_unig

    firstWordKey_big3=big3.map(firstKeyReformat)

    groupedByFirstWord=firstWordKey_big3.leftOuterJoin(unig3).map(groupedReformat1)
    groupedBySecondWord=groupedByFirstWord.leftOuterJoin(unig3).map(groupedReformat2)
    P=groupedBySecondWord.map(partial(calculateP, B, U, fg_total_big, bg_total_big, fg_total_unig))

    score=P.map(partial(calculateInfoPhrase, w_info, w_phrase))

    sorted_score=sorted(score.collect(), key=getkey, reverse=True)
    for i in range(0, n_outputs):
        print str(sorted_score[i][0])+":"+str(sorted_score[i][1])
    """ terminate """
    sc.stop()


if __name__ == '__main__':
    main(sys.argv)

