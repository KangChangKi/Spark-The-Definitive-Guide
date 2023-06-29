from pyspark.sql import SparkSession
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors

denseVec = Vectors.dense(1.0, 2.0, 3.0)
size = 3
idx = [1, 2]  # locations of non-zero elements in vector
values = [2.0, 3.0]
sparseVec = Vectors.sparse(size, idx, values)

# project-templates/python/pyspark_template/main.py

spark = SparkSession.builder \
    .master("local") \
    .appName("Test") \
    .getOrCreate()

# 데이터 준비

# df = spark.read.json("/data/simple-ml")
df = spark.read.json(
    "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide/data/simple-ml")
df.orderBy("value2").show()


# weights 준비

supervised = RFormula(formula="lab ~ . + color:value1 + color:value2")


# 데이터를 원하는 형태로 변경시킴

fittedRF = supervised.fit(df)
preparedDF = fittedRF.transform(df)
# column "label" (output) 과 column "features" (input) 가 새로 생성됨.
preparedDF.show(truncate=False)


# data 를 train set 과 test set 으로 분리

train, test = preparedDF.randomSplit([0.7, 0.3])


# 모델 준비

lr = LogisticRegression(labelCol="label", featuresCol="features")


# 모델의 parameters 의 설명 출력

print(lr.explainParams())


# train 시킴

fittedLR = lr.fit(train)

# prediction 수행
res = fittedLR.transform(test)
res.show(truncate=False)
res.createOrReplaceTempView("res")
spark.sql("select prediction from res").show()

#################################################################


# COMMAND ----------

train, test = df.randomSplit([0.7, 0.3])


# COMMAND ----------

rForm = RFormula()
lr = LogisticRegression().setLabelCol("label").setFeaturesCol("features")


# COMMAND ----------

stages = [rForm, lr]
pipeline = Pipeline().setStages(stages)


# COMMAND ----------

params = ParamGridBuilder()\
    .addGrid(rForm.formula, [
        "lab ~ . + color:value1",
        "lab ~ . + color:value1 + color:value2"])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .addGrid(lr.regParam, [0.1, 2.0])\
    .build()


# COMMAND ----------

evaluator = BinaryClassificationEvaluator()\
    .setMetricName("areaUnderROC")\
    .setRawPredictionCol("prediction")\
    .setLabelCol("label")


# COMMAND ----------

tvs = TrainValidationSplit()\
    .setTrainRatio(0.75)\
    .setEstimatorParamMaps(params)\
    .setEstimator(pipeline)\
    .setEvaluator(evaluator)


# COMMAND ----------

tvsFitted = tvs.fit(train)


# COMMAND ----------
