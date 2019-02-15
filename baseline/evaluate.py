#!/usr/bin/env python
import sys, os, os.path
import numpy as np


input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

print(submit_dir, truth_dir)

if not os.path.isdir(submit_dir):
	print("%s doesn't exist" % submit_dir)

if not os.path.isdir(truth_dir):
	print("%s doesn't exist" % truth_dir)

# if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

submission = open(os.path.join(submit_dir, "answer.tsv"), "r")
reference = open(os.path.join(truth_dir, "test.tsv"), "r")


print("files opened")

    # submission = open("answer.tsv", "r").readlines()
    # reference = open("reference.tsv", "r").readlines()
    # assert len(submission) == len(reference), "no. of lines in submission file does not match the same in ground truth file"

preds = dict()
truths=dict() # Dictionary with key = query_id and value = array of labels for respective passages
passages=dict()
queries=dict()
for sub in submission:
    sub = list(map(float, sub.strip("\n").split("\t")))
    preds[int(sub[0])] = sub[1:]

for ref in reference:
    ref = ref.strip("\n").split("\t")
    query_id=int(ref[0])
    label=int(ref[3])

    passage=str(ref[2])
    query=str(ref[1])

    if(query_id in truths):
        truths[query_id].append( label )
        passages[query_id].append(passage)
    else:
        truths[query_id] = [ label ]
        passages[query_id]=[passage]
        queries[query_id]=query


# for ref in reference:
#     ref = list(map(int, ref.strip("\n").split("\t")))
#     truths[int(ref[0])] = ref[1:]


scores = []

output_filename = os.path.join(output_dir, 'bad_results_rank34.txt')              
output_file = open(output_filename, 'w')


for q_id in truths:
    if q_id not in preds:
        print("Query not in prediction")
        scores.append(0)
    else:
        # print("Query found!")
        selected_psg = np.nonzero(truths[q_id])[0][0]
        # print("Selected passage: ",selected_psg)
        print(selected_psg)
        sorted_preds = np.argsort(preds[q_id])[::-1]
        rank = np.where(sorted_preds==selected_psg)[0][0] + 1
        print("Rank= {}, Query_id= {}".format(rank,q_id) )
        scores.append(1.0/rank)
        if(rank>=3 and rank<=4):
            output_file.write("\nRank= {}, Query_id= {}\n".format(rank,q_id) )        
            # print(sorted_preds)
            output_file.write("Query: "+str(queries[q_id])+"\n\n" )
            for i in range(rank):
                output_file.write("Score: "+str([preds[q_id][sorted_preds[i]] ]))
                output_file.write(passages[q_id][sorted_preds[i]]+'\n\n' )
           
score = np.mean(scores)
print(score)

output_filename = os.path.join(output_dir, 'scores.txt')              
output_file = open(output_filename, 'w')
output_file.write("Difference: %f" % score)
output_file.close()