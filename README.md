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


TF-IDF

| ML algorithm  | accuracy | time taken (s)  |
|---------------|----------|---------|
|KNN            | 0.76     | 3.90    |
|SVM            | 0.87     | 3.04    |
|NN             | 0.88     | 71.00   |
|naiveBayes     | 0.85     | 1.66    |

TF-IDF SVD

| ML algorithm  | accuracy | time taken (s)  |
|---------------|----------|---------|
|KnnSvd(1000)   | 0.77     | 451.80  |
|KnnSvd(2000)   | 0.63     | 244.79  |
|SvmSvd(1000)   | 0.84     | 34.86   |
|SvmSvd(2000)   | 0.86     | 93.79   |
|NnSvd (2000)   | 0.85     | 95.01   |

TF-IDF reduced

| ML algorithm  | accuracy | time taken |
|---------------|----------|------------|
|KNN            | 0.76     |     |
|SVM            | 0.87     |  |
|NN             | 0.88     |   |
|naiveBayes     | 0.85     |   |
|KnnSvd         | 0.77     | 759.50  |
|SvmSvd         | 0.87     |   |
|NnSvd (3)      | 0.35(8s) |   |




term frequency

| ML algorithm  | accuracy | time taken  |
|---------------|----------|---------|
|KNN            | 0.28     | 4.13    |
|SVM            | 0.81     | 5.57    |
|NN             | 0.86     | 64.08   |
|naiveBayes     | 0.81     | 1.75    |

term frequency with SVD

| ML algorithm  | accuracy | time taken  |
|---------------|----------|--------|
|KnnSvd(2000)   | 0.38     | 207.27 |
|KnnSvd(2000)   | 0.38     | 232.90 |
|SvmSvd(2000)   | 0.79     | 163.63 |
|NnSvd (2000)   | 0.79     | 85.75  |

* SVD cannot be executed with more than 8k features, out of memory error

term frequency reduced

| ML algorithm  | accuracy    | time taken |
|---------------|----------|------------|
|KNN            | 0.34 (1000) |   |
|SVM            | 0.83  (7)   |   |
|NN             | 0.87  (10)  |   |
|naiveBayes     | 0.84  (7)   |   |


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

The experiment:

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
6 tfidf with naive reduction (not tested yet)
 