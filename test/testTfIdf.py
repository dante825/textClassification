from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, IndexToString, HashingTF, IDF
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
# inputFileLoc = '/home/dante/development/workspaces/pycharm-workspace/textClassification/output/test.csv'

spark = SparkSession.builder \
            .appName('TEXT_CLASSIFICATION') \
            .master('local[*]') \
            .config('spark.driver.memory', '4G') \
            .config('spark.cores.max', '4') \
            .getOrCreate()

spark.sparkContext.setLogLevel("OFF")

logging.info('Reading 20 news group datasets.')
data = spark.read.csv(inputFileLoc, header=True)

# data = spark.createDataFrame([
#     (0, "Hi I heard about Spark", "a"),
#     (1, "I wish Java could use case classes", "b"),
#     (2, "Logistic,regression,models,are,neat", "a")
# ], ["id", "text", "category"])

# Displaying the data read
data.show()

tokenizer = Tokenizer(inputCol='text', outputCol='words')
wordsData = tokenizer.transform(data)

hashingTf = HashingTF(inputCol='words', outputCol='rawFeatures', numFeatures=20)
featurizedData = hashingTf.transform(wordsData)

idf = IDF(inputCol='rawFeatures', outputCol='features')
idfModel = idf.fit(featurizedData)
rescaledData =  idfModel.transform(featurizedData)

rescaledData.select('category', 'features').show()