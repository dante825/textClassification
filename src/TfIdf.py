from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, IndexToString, HashingTF, IDF
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
import logging

logging.basicConfig(level=logging.INFO)

# inputFileLoc = '/home/db/development/workspaces/pycharmWorkspace/textClassification/output/20newsGroup.csv'
inputFileLoc = '/home/dante/development/workspaces/pycharm-workspace/textClassification/output/20newsGroup.csv'

spark = SparkSession.builder \
            .appName('TEXT_CLASSIFICATION') \
            .master('local[*]') \
            .config('spark.driver.memory', '4G') \
            .config('spark.cores.max', '4') \
            .getOrCreate()

spark.sparkContext.setLogLevel("OFF")

logging.info('Reading 20 news group datasets.')
data = spark.read.csv(inputFileLoc, header=True)

# Displaying the data read
# data.show()

tokenizer = Tokenizer(inputCol='text', outputCol='words')
wordsData = tokenizer.transform(data)

hashingTf = HashingTF(inputCol='words', outputCol='rawFeatures', numFeatures=20)
featurizedData = hashingTf.transform(wordsData)

idf = IDF(inputCol='rawFeatures', outputCol='features')
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

indexer = StringIndexer(inputCol='category', outputCol='label')
model = indexer.fit(rescaledData)
indexed = model.transform(rescaledData)

# indexed.show()
# indexed.select('category', 'features', 'label').show()
# To know which category is the index
indexed.groupBy(["category", 'label']).count().sort('label').show(truncate=False)

# Partition the data for training and testing
(trainingData, testData) = indexed.randomSplit([0.7, 0.3], seed=100)
print("training dataset count: " + str(trainingData.count()))
print("testing dataset count: " + str(testData.count()))

#### Linear Support Vector Machine
# Only support binary classification! Pyspark don't have the 3 ML algorithms needed!
lsvc = LinearSVC(maxIter=10, regParam=0.1)
lsvcModel = lsvc.fit(trainingData)
print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))