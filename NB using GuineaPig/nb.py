from guineapig import *
import sys
import math
import logging

def tokens(lw):
	for label in lw[0]:
		for w in lw[1]:
			yield (w,label)

def tokensTest(lw):
	for w in lw[1]:
		yield (w,lw[0])

def predictClass(p1, p2):
	docid=p1[0]
	words=p1[1]
	classes=p2[0]
	qx=p2[1]


	domY=len(classes)
	anyY=0
	for key in classes.keys():
		anyY=anyY+classes[key]


	bestLabel=''
	bestprob=-1000000000
	for label in classes.keys():
		prob=math.log((classes[label]+(1.0/domY))/(anyY+1.0))
		for word in words:
			if label in word.keys():
				prob=prob+math.log((word[label]+qx)/(classes[label]+1.0))
			else:
				prob=prob+math.log(qx/(classes[label]+1.0))
		#print label, prob
		if prob>bestprob:
			bestprob=prob
			bestLabel=label
	yield(docid, bestLabel, bestprob)



class NB(Planner):

	params = GPig.getArgvParams()
	data = ReadLines(params['trainFile']) \
		| Map(by=lambda line:line.strip().split("\t"))\
		| Map(by=lambda (docid, label, doc) : ( label.split(","), (doc.lower()).split()))\
		| Flatten( by=tokens)

	wordCount = Group(data, by=lambda x:x, reducingTo=ReduceToCount())

	V=Group(data, by=lambda x:"hhhhh", reducingTo=ReduceToCount())\
		|ReplaceEach(by=lambda (x,y):1.0/y)


	numY=Group(data, by=lambda (word,label):label,retaining=lambda (word, label):label,  reducingTo=ReduceToCount())


	dictY= Group(numY, by=lambda x:"c", reducingTo=ReduceToList())\
		| Map(by=lambda (x,y): dict(y))


	table1=Group(wordCount, by=lambda ((word,label),num):word, retaining=lambda ((word,label),num): (label,num),  reducingTo=ReduceToList())

	dataTest= ReadLines(params['testFile']) \
		| Map(by=lambda line:line.strip().split("\t"))\
		| Map( by=lambda (docid, label, doc) : ( docid, (doc.lower()).split())) \
		| Flatten(by=tokensTest)

	table2=Join( Jin(table1, by=lambda (word, value): word), Jin(dataTest, by=lambda (word, id): word)) \
		| Map(by=lambda ((word1, value),(word2, id)): ( dict(value), id))\
		| Group(by=lambda (value, id): id, retaining=lambda (value, id):value, reducingTo=ReduceToList())
	table3 = Augment(table2, sideviews=[dictY, V], loadedBy=lambda v1, v3: (GPig.onlyRowOf(v1), GPig.onlyRowOf(v3)) ) 

	output=Flatten(table3, by=lambda (p1, p2):predictClass(p1,p2))


# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)