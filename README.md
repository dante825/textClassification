# textClassification
Text classification with 20 news group dataset

The dataset is available at: http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

Dataset categories count

using preprocessing label encoder

|category                 |category_id   |count|
|-------------------------|--------------|-----|  
|alt.atheism              |0             |798  |
|comp.graphics            |1             |971  |
|comp.os.ms-windows.misc  |2             |980  |
|comp.sys.ibm.pc.hardware |3             |979  |
|comp.sys.mac.hardware    |4             |957  |
|comp.windows.x           |5             |978  |
|misc.forsale             |6             |962  |
|rec.autos                |7             |988  |
|rec.motorcycles          |8             |993  |
|rec.sport.baseball       |9             |993  |
|rec.sport.hockey         |10            |998  |
|sci.crypt                |11            |991  |
|sci.electronics          |12            |981  |
|sci.med                  |13            |988  |
|sci.space                |14            |986  |
|soc.religion.christian   |15            |997  |
|talk.politics.guns       |16            |910  |
|talk.politics.mideast    |17            |940  |
|talk.politics.misc       |18            |775  |
|talk.religion.misc       |19            |628  |


Test results are in the output/results.ods

Dimension reduction would reduce the accuracy of the classification because it remove
the information available for the classifier.

Lemmatization does not have any improvement on the accuracy compared to stemming

Report outline
1. changes in literature review
    - include tf in feature extraction part
    - in dimension reduction, find proof that SVD is better, PCA is not that suitable for sparse matrix
2. in research methodology, more details on preprocessing and include more details on the
   steps taken.
3. results of the experiments
4. discussion of the results
5. conclusion

The experiments:

Since the KNN article used TF for feature extraction and performed vector space reduction on it, and
accuracy increased after vector space reduction. The experiment would be to verify that and test it out
with other machine learning algorithm. 
Beside only using the TF mentioned, TF-IDF feature extraction method is explored and
SVD dimension reduction is applied on the matrix
1. With term frequency
2. with term frequency and naive reduction
3. with term frequency and SVD
4. tfidf
5. tfidf with SVD
6. tfidf with naive reduction

maybe add some graphs or picture before results and discussion,
pictures showing the categories distribution of the data
 
