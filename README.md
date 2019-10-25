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


Accuracy comparison on different machines (limit tf-idf features to 8000)

| ML algorithm  | home-pc  | work-pc | unlimited to 8k |
|---------------|----------|---------|-----------------|
|KNN            | 0.36     |   0.36  |  0.87           |
|SVM            | 0.88     |   0.88  |  0.91           |
|NN             | 0.88(56s)|0.88(57s)|  0.92           |
|naiveBayes     | 0.84     |   0.84  |
|KnnSvd (5)     | 0.23(3s) |0.23(2s) |
|SvmSvd(4)      | 0.33(3s) |0.33(3s) |
|NnSvd (3)      | 0.35(8s) |0.34(5s) |

Dimension reduction would reduce the accuracy of the classification because it remove
the information available for the classifier.

Have to find some methods to increase the accuracy.
- Standard scaler
- lemmatization