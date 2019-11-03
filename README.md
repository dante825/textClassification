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


Accracy scores with lemmatization

| ML algorithm  | 8000     | \> 8k   |
|---------------|----------|---------|
|KNN            | 0.76     |  0.76   |
|SVM            | 0.88     |  0.91   |
|NN             | 0.88     |  0.91   |
|naiveBayes     | 0.84     |  0.84   |
|KnnSvd (3)     | 0.35(3s) |  0.13   |
|SvmSvd(4)      | 0.33(3s) |  0.13   |
|NnSvd (3)      | 0.35(8s) |  0.13   |

Accuray scores with term frequency

| ML algorithm  | 8000     | \> 8k    |
|---------------|----------|---------|
|KNN            | 0.28     |     |
|SVM            | 0.81     |     |
|NN             | 0.84     |     |
|naiveBayes     | 0.81     |     |
|KnnSvd         | 0.31 |     |
|SvmSvd         | 0.80 |     |
|NnSvd          | 0.80 |     |


Dimension reduction would reduce the accuracy of the classification because it remove
the information available for the classifier.

Lemmatization does not have any improvement on the accuracy compared to stemming

ways to explore to solve this problem:
1. according to the KNN term reduction paper, seems like it is using term frequency rather than tf-idf, it deletes the term that appeared only once.
2. with term frequency then dimension reduction should have an increase in accuracy.
3. has problem to remove the columns of features with term frequency 1 