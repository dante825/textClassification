# textClassification
Text classification with 20 news group dataset

The dataset is available at: http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

Dataset categories count

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


Roadmap or milestones:
- Knn [working, 0.80]
- SVM [linearSVC 0.91]
- Neural network [sort of working, accuracy dependent on training time, 0.92]

- Knn with sparse PCA or truncated SVD [not working]
- SVM with sparse PCA or truncated SVD [not working]
- NN with sparse PCA or truncated SVD [not working]