# MS AI Challenge
Team:Sushant Rathi, Shreyas Singh and I(Aditya Malte) 
Dataset: Microsoft AI Challenge 2018 dataset as available on CodaLab
Algorithm: Bidirectional Encoder Representation From Transformer(BERT)
Model: BERT_BASE


The Microsoft AI Challenge was a competition held by Microsoft to improve their Bing search results and make them more succinct.
As part of the MSAI Challenge, given a question query and 10 passages, our model has to choose the answer that most aptly answers the question.

Our approach:
1) Initially perform smart downsample to balance the unbalanced dataset:
We first performed undersampling of the dataset such that our dataset consisted only of those passages that were closest to the ground truth passage. We used Okapi-BM25 to perform this undersampling, keeping the incorrect to correct ratio as 2:1.
2) We then fine-tuned on BERT using its next-sentence prediction algorithm. To perform this task, we modified the MRPC class of BERT to suit our dataset. 
3) In addition, we also added the functionality of class weights to account for the unbalanced dataset.

Results:
Our model achieved impressive results of F1 0.675 in Phase 1 of the MSAI challenge(Phase 2 score is undisclosed by Microsoft). Thus, placing us in the top 1% of the leaderboard of 2000 team in the MSAI Challenge.
