from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
import logging

logging.basicConfig(level=logging.INFO)

inputFileLoc = '/home/db/development/workspaces/pycharmWorkspace/textClassification/output/20newsGroup.csv'
# inputFileLoc = '/home/dante/development/workspaces/pycharm-workspace/textClassification/output/20newsGroup.csv'

spark = SparkSession.builder \
            .appName('TEXT_CLASSIFICATION') \
            .master('local[*]') \
            .config('spark.driver.memory', '4G') \
            .config('spark.cores.max', '4') \
            .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

logging.info('Reading 20 news group datasets.')
# data = spark.read.csv(inputFileLoc, header=True)

data = spark.createDataFrame([
    (0, "Hi I heard about Spark", "a"),
    (1, "I wish Java could use case classes", "b"),
    (2, "Logistic,regression,models,are,neat", "a")
], ["id", "text", "category"])

# Displaying the data read
data.show()
# data.groupBy("category").count().show()

# Creating a pipeline
###### Tokenizer ########
# 2 methods, tokenizer or regexTokenizer, same output
# tokenizer = Tokenizer(inputCol='text', outputCol='words')
regexTokenizer = RegexTokenizer(inputCol='text', outputCol='words', pattern='\\W')
countTokens = udf(lambda words: len(words), IntegerType())

# tokenized = tokenizer.transform(data)
# tokenized.select('text', 'words').withColumn('tokens', countTokens(F.col('words'))).show(truncate=False)

regexTokenized = regexTokenizer.transform(data)
regexTokenizedData = regexTokenized.select('text', 'words', 'category').withColumn('tokens', countTokens(F.col('words')))
regexTokenizedData.show(truncate=False)

########## Stopwords Remover #########
stopwordsRemover = StopWordsRemover(inputCol='words', outputCol='filtered')
removedStopWordsData = stopwordsRemover.transform(regexTokenizedData)
removedStopWordsData.show(truncate=False)

# countVectors = CountVectorizer(inputCol='filtered', outputCol='features', vocabSize=3, minDF=2.0)
countVectors = CountVectorizer(inputCol='filtered', outputCol='features')
model = countVectors.fit(removedStopWordsData)
result = model.transform(removedStopWordsData)
result.show(truncate=False)

######## String indexer #########
indexer = StringIndexer(inputCol='category', outputCol='label')
model = indexer.fit(result)
indexed = model.transform(result)

print("Transformed string column '%s' to indexed column '%s'"
      % (indexer.getInputCol(), indexer.getOutputCol()))
indexed.show()

print("StringIndexer will store labels in output column metadata\n")

converter = IndexToString(inputCol="label", outputCol="originalCategory")
converted = converter.transform(indexed)

print("Transformed indexed column '%s' back to original string column '%s' using "
      "labels in metadata" % (converter.getInputCol(), converter.getOutputCol()))
converted.select("label", "originalCategory").show()

##### Pipeline #########
logging.info('Using all the components in a pipeline')
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, indexer])

# Fit the pipeline to training documents
pipelineModel = pipeline.fit(data)
pipelineResult = pipelineModel.transform(data)
pipelineResult.show()
