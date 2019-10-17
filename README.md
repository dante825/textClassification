# textClassification
Text classification with 20 news group dataset

The dataset is available at: http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

Dataset categories count
using df.factorize

|category                |label|count|
|------------------------|-----|-----|
|rec.sport.baseball      |0    |993  |
|sci.med                 |1    |988  |
|comp.sys.mac.hardware   |2    |957  |
|talk.politics.mideast   |3    |940  |
|comp.graphics           |4    |970  |
|misc.forsale            |5    |962  |
|talk.politics.guns      |6    |910  |
|rec.motorcycles         |7    |993  |
|sci.electronics         |8    |981  |
|comp.windows.x          |9    |978  |
|rec.autos               |10   |988  |
|talk.politics.misc      |11   |775  |
|comp.os.ms-windows.misc |12   |980  |
|talk.religion.misc      |13   |628  |
|sci.space               |14   |986  |
|rec.sport.hockey        |15   |998  |
|soc.religion.christian  |16   |997  |
|alt.atheism             |17   |798  |
|comp.sys.ibm.pc.hardware|18   |979  |
|sci.crypt               |19   |991  |

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

Roadmap or milestones:
- naiveBayes [0.87]
- Knn [working, 0.83]
- SVM [linearSVC 0.91]
- Neural network [sort of working, accuracy dependent on training time, 0.92]

- Knn with sparse PCA or truncated SVD [not working]
- SVM with sparse PCA or truncated SVD [not working]
- NN with sparse PCA or truncated SVD [not working]

Sparse PCA ran out of memory and Truncated SVD accuracy is bad (0.06)\
Back to TF-IDF, add stemmer or lemmatization, remove stop words

Stemming may reduce the dimension\
Currently, after tfidf the dimension is aorund (15033,130877)\
After stop words removal, 125026
After stemming: 101790