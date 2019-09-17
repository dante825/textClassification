from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
import logging

logging.basicConfig(level=logging.INFO)

spark = SparkSession.builder \
            .appName('TEXT_CLASSIFICATION') \
            .master('local[*]') \
            .config('spark.driver.memory', '4G') \
            .config('spark.cores.max', '4') \
            .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

logging.info('Reading 20 news group datasets.')
# data = spark.read.csv('/home/dante/development/workspaces/pycharm-workspace/textClassification/output/20newsGroup.csv',
#                       header=True)

data = spark.createDataFrame([
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["id", "text"])

# Displaying the data read
data.show()
# data.groupBy("category").count().show()

# Creating a model pipeline
###### Tokenizer ########
# tokenizer = Tokenizer(inputCol='text', outputCol='words')
regexTokenizer = RegexTokenizer(inputCol='text', outputCol='words', pattern='\\W')
countTokens = udf(lambda words: len(words), IntegerType())

# tokenized = tokenizer.transform(data)
# tokenized.select('text', 'words').withColumn('tokens', countTokens(F.col('words'))).show(truncate=False)

regexTokenized = regexTokenizer.transform(data)
regexTokenized.select('text', 'words').withColumn('tokens', countTokens(F.col('words'))).show(truncate=False)

# addStopwords = ['http', 'https', 'amp', 'rt', 't', 'c', 'the']

# stopwordsRemover = StopWordsRemover(inputCol='words', outputCol='filtered')
# stopwordsRemover.transform(data).show(truncate=False)

# countVectors = CountVectorizer(inputCol='filtered', outputCol='features', vocabSize=10000, minDF=5)

# String indexer
# label_stringIdx = StringIndexer(inputCol='category', outputCol='label')

# pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])

# Fit the pipeline to training documents
# pipelineModel = pipeline.fit(data)
# result = pipelineModel.transform(data)
# result.show()
