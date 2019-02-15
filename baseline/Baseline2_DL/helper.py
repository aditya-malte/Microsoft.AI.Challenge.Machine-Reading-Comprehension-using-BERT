import os

#Some useful methods


def split_tsv(inputFile, trainFile, validationFile,split=0.9):

    numQueries=12000 #limit number of queries to be read from data.tsv (max 524188)
    start=36000
    samples_per_query=3
    skip=start*samples_per_query
 
    f = open(inputFile,"r",encoding="utf-8",errors="ignore")  # Format of the file : query_id \t query \t passage \t label \t passage_id
    fw_train = open(trainFile,"w",encoding="utf-8")
    fw_validation= open(validationFile,"w",encoding="utf-8")

    trainSetSize=(int(split*numQueries))
    print("Size of train,validation: ",trainSetSize, numQueries-trainSetSize)
    count=0
    for line in f:
        count+=1
        print(count)
        if(count<skip):
            continue
        if(count<=skip + trainSetSize*samples_per_query):
            fw_train.write(line)
        elif(count<=skip + numQueries*samples_per_query):
            fw_validation.write(line)
        else: 
            break
    
    f.close()
    fw_train.close()
    fw_validation.close()

