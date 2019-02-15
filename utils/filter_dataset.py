import os
import math
import pickle
import sys

from preprocess import tokenize

import random
# from random import randint


#Initialize Global variables 
docIDFDict = {}
avgDocLength = 0


def univEncsimilarity(passage1, passage2):
    # Return semantic similarity of passage1 and passage2 based on universal sentence encoder embeddings
    pass

#The following GetBM25Score method will take 2 passages as input and outputs their similarity score based on the term frequency(TF) and IDF values.
def BM25Similarity(Query, Passage, k1=1.5, b=0.75, delimiter=' ') :
    
    global docIDFDict,avgDocLength

    # query_words= Query.strip().lower().split(delimiter)
    # passage_words = Passage.strip().lower().split(delimiter)
    query_words=tokenize(Query.strip().lower())
    passage_words = tokenize(Passage.strip().lower())
    passageLen = len(passage_words)
    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :   
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula 
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score


def filter_dataset(inputFile,outputFile,simFunction):
    # Reduce number of negative samples, keep 2 which are most similar to the positive one
    f = open(inputFile,'r',encoding="utf-8")
    fw = open(outputFile,'w',encoding="utf-8")    
    sameQuerySet=[]  #This will store negative entries belonging to the same query
    lno=0
    start=1000000
    num_samples=500000
    for line in f:
        lno+=1        
        if(lno<=start):
            continue
        tokens = line.strip().lower().split("\t")
        if(tokens[3]=="1"):
            # print("Correct Passage")
            Query = tokens[1]
            Passage = tokens[2]
            correct=tokens
        else: 
            sameQuerySet.append(tokens)
        # print(lno)
        if(lno==start+num_samples+1):
            break
        if(lno%10==0):
            simScores=[]
    
            if(random.uniform(0,1)<=1.0): # In 100 % cases, pick examples most similar to the correct one
                for i in range(len(sameQuerySet)):
                    ithPassage=sameQuerySet[i][2] #Passage at ith index
                    simScores.append( (i, simFunction(ithPassage,Passage) ) ) #append score
                sortedSimScores=sorted(simScores,key=lambda x: x[1])

                best=sameQuerySet[sortedSimScores[-1][0]]
                secondbest=sameQuerySet[sortedSimScores[-2][0]]
            else:
                a,b=random.sample(range(0, 8), k=2)
                best=sameQuerySet[a]
                secondbest=sameQuerySet[b]
                

            fw.write("\t".join(correct)+"\n")
            fw.write("\t".join(best)+"\n")
            fw.write("\t".join(secondbest)+"\n")

            if(lno%10000==0):
                print("## Correct passage: ",Passage)
                print("## Most similar: ",best[2])
                print("## SecondMost similar: ",secondbest[2])
            
            
            sameQuerySet=[]
        if(lno%5000==0):
            print(lno)
    
    f.close()
    fw.close()


if __name__ == "__main__":
    # global docIDFDict,avgDocLength

    inputFile=sys.argv[1]
    outputFile=sys.argv[2]
    simFunction=sys.argv[3] #0 for BM25 based on tf-idf, 1 for universal sentence encoder
    print(simFunction)
    if(simFunction=='0'):
        outputFile='BM25_'+outputFile
        print('BM25 based')
        with open('../baseline/Baseline1_BM25/docIDFDict.pickle', 'rb') as handle:
            docIDFDict = pickle.load(handle)
            avgDocLength=56
        filter_dataset(inputFile,outputFile,BM25Similarity)
    else:
        print('universal encoder based')        
        outputFile='univenc_'+outputFile
        filter_dataset(inputFile,outputFile,univEncsimilarity)
        