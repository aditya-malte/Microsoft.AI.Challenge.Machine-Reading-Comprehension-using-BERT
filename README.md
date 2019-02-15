# Microsoft AI Challenge(Top 20)
Team:Sushant Rathi, Shreyas Singh and I(Aditya Malte) 

Dataset: Microsoft AI Challenge 2018 dataset as available on CodaLab

Algorithm: Bidirectional Encoder Representation From Transformer(BERT)

Model: BERT_BASE


The Microsoft AI Challenge was a competition held by Microsoft to improve their Bing search results and make them more succinct.
As part of the MSAI Challenge, given a question query and 10 passages, our model has to choose the answer that most aptly answers the question.

Our approach:
1) Initially perform smart downsample to balance the unbalanced dataset:
We first performed undersampling of the dataset such that our dataset consisted only of those passages that were closest to the ground truth passage. We used Okapi-BM25 to perform this undersampling, keeping the incorrect to correct ratio as 2:1.
2) We then fine-tuned on BERT using its next-sentence prediction algorithm. To perform this task, we modified the MRPC class of BERT to suit our dataset. We trained this as a binary classification task. Finally, the highest ranking passage is the one with highest probability. 
3) In addition, we also added the functionality of class weights to account for the unbalanced dataset.

Results:
Our model achieved impressive results of F1 0.6715 in Phase 1 of the MSAI challenge(Phase 2 score is undisclosed by Microsoft). Thus, placing us in the top 1% of the leaderboard of 2000 team in the MSAI Challenge despite the fact that we had to stop training(on a fraction of the whole data) before convergence due to exhaustion of computation power.

Future scope:
1)Use boosting/ensembling when more processing becomes available.
2)Try more architectures for the final layer.
3)Train on more epochs and data.
4)Try training as a multi-classification task.

