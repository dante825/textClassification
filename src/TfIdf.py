from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import logging

logging.basicConfig(level=logging.INFO)

# inputFileLoc = '/home/db/development/workspaces/pycharmWorkspace/textClassification/output/20newsGroup.csv'
inputFileLoc = '/home/dante/development/workspaces/pycharm-workspace/textClassification/output/20newsGroup.csv'
trainOutputLoc = '/home/dante/development/workspaces/pycharm-workspace/textClassification/output/train.csv'
testOutputLoc = '/home/dante/development/workspaces/pycharm-workspace/textClassification/output/test.csv'

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
# indexed.groupBy(["category", 'label']).count().sort('label').show(truncate=False)

# Partition the data for training and testing
(trainingData, testData) = indexed.randomSplit([0.7, 0.3], seed=100)
# print("training dataset count: " + str(trainingData.count())) #14026
# print("testing dataset count: " + str(testData.count())) #5936

train_df = trainingData.toPandas()
sub_train_df = train_df[['features', 'label']]
sub_train_df.to_csv(trainOutputLoc, index=False)
# print(sub_train_df.head(10))
# print(sub_train_df.columns.values)
test_df = testData.toPandas()
sub_test_df = test_df[['features', 'label']]
# print(sub_test_df.head(10))
sub_test_df.to_csv(testOutputLoc, index=False)
