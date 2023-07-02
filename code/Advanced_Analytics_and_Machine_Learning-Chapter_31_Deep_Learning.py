from pyspark.sql import SparkSession

# 참고1: https://spark-packages.org/package/databricks/spark-deep-learning
# 
# Version: 1.5.0-spark2.4-s_2.11 ( 154e7a | zip | jar ) / Date: 2019-01-25 / License: Apache-2.0 / Scala version: 2.11
# Version: 1.4.0-spark2.4-s_2.11 ( a4792f | zip | jar ) / Date: 2018-11-18 / License: Apache-2.0 / Scala version: 2.11
# Version: 1.3.0-spark2.4-s_2.11 ( 7e5238 | zip | jar ) / Date: 2018-11-12 / License: Apache-2.0 / Scala version: 2.11
# ...
# 안타깝게도 Spark 3 에 대응하는 version 이 나오지 않은 것 같다.
# 아마도 개발이 중지된 것 같다.
# 그래서 `sparkdl` package 를 사용할 수 없어서, 아래의 코드를 사용할 수가 없다.
# 
# 현재 Spark 에서 DeepLearning 을 주도하는 라이브러리는 H2O 인 것 같다.
# 일단 Databricks 는 주도권을 잃은 것 처럼 보인다.
# 대신 H2O 가 이 분야를 주도하는 것 처럼 보인다.

spark = (SparkSession.builder.appName("test1").master("local")
  # .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")
  .getOrCreate())

path_prefix = "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide"

from sparkdl import readImages

img_dir = path_prefix + '/data/deep-learning-images/'
image_df = readImages(img_dir)

image_df.printSchema()

#########################################################################################################

# COMMAND ----------

from sparkdl import readImages
from pyspark.sql.functions import lit

tulips_df = readImages(img_dir + "/tulips").withColumn("label", lit(1))
daisy_df = readImages(img_dir + "/daisy").withColumn("label", lit(0))
tulips_train, tulips_test = tulips_df.randomSplit([0.6, 0.4])
daisy_train, daisy_test = daisy_df.randomSplit([0.6, 0.4])
train_df = tulips_train.unionAll(daisy_train)
test_df = tulips_test.unionAll(daisy_test)


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features",
  modelName="InceptionV3")
lr = LogisticRegression(maxIter=1, regParam=0.05, elasticNetParam=0.3,
  labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

tested_df = p_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(tested_df.select(
  "prediction", "label"))))


# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import expr

# a simple UDF to convert the value to a double
def _p1(v):
  return float(v.array[1])

p1 = udf(_p1, DoubleType())
df = tested_df.withColumn("p_1", p1(tested_df.probability))
wrong_df = df.orderBy(expr("abs(p_1 - label)"), ascending=False)
wrong_df.select("filePath", "p_1", "label").limit(10).show()


# COMMAND ----------

from sparkdl import readImages, DeepImagePredictor

image_df = readImages(img_dir)
predictor = DeepImagePredictor(
  inputCol="image",
  outputCol="predicted_labels",
  modelName="InceptionV3",
  decodePredictions=True,
  topK=10)
predictions_df = predictor.transform(image_df)


# COMMAND ----------

df = p_model.transform(image_df)


# COMMAND ----------

from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF
from keras.applications import InceptionV3

registerKerasImageUDF("my_keras_inception_udf", InceptionV3(weights="imagenet"))


# COMMAND ----------

