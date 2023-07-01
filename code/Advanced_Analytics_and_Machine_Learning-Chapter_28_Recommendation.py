from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test1").master("local").getOrCreate()

path_prefix = "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide"

from pyspark.sql import Row
ratings = spark.read.text(path_prefix + "/data/sample_movielens_ratings.txt")\
  .selectExpr("split(value , '::') as col")\
  .selectExpr(
    "cast(col[0] as int) as userId",
    "cast(col[1] as int) as movieId",
    "cast(col[2] as float) as rating",
    "cast(col[3] as long) as timestamp")

ratings.dtypes

# [('userId', 'int'),
#  ('movieId', 'int'),
#  ('rating', 'float'),
#  ('timestamp', 'bigint')]

training, test = ratings.randomSplit([0.8, 0.2])

##########################################################################################################

# ALS 는 아이템의 특징 벡터와 사용자의 특징 벡터의 내적이 사용자의 평점 벡터와 유사하도록 만드는 특징 벡터를 찾는다.
# 즉, v_item . v_user . v_mapping ~= v_rating 인 v_mapping 을 찾는다.

# Spark 에서 ALS 를 미는 이유는 대규모 처리가 가능하기 때문이다.

from pyspark.ml.recommendation import ALS

als = ALS()\
  .setMaxIter(5)\
  .setRegParam(0.01)\
  .setUserCol("userId")\
  .setItemCol("movieId")\
  .setRatingCol("rating")
print(als.explainParams())

# alpha: alpha for implicit preference (default: 1.0)
# blockSize: block size for stacking input data in matrices. Data is stacked within partitions. If block size is more than remaining data in a partition then it is adjusted to the size of this data. (default: 4096)
# checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext. (default: 10)
# coldStartStrategy: strategy for dealing with unknown or new users/items at prediction time. This may be useful in cross-validation or production scenarios, for handling user/item ids the model has not seen in the training data. Supported values: 'nan', 'drop'. (default: nan)
# finalStorageLevel: StorageLevel for ALS model factors. (default: MEMORY_AND_DISK)
# implicitPrefs: whether to use implicit preference (default: False)
# intermediateStorageLevel: StorageLevel for intermediate datasets. Cannot be 'NONE'. (default: MEMORY_AND_DISK)
# itemCol: column name for item ids. Ids must be within the integer value range. (default: item, current: movieId)
# maxIter: max number of iterations (>= 0). (default: 10, current: 5)
# nonnegative: whether to use nonnegative constraint for least squares (default: False)
# numItemBlocks: number of item blocks (default: 10)
# numUserBlocks: number of user blocks (default: 10)
# predictionCol: prediction column name. (default: prediction)
# rank: rank of the factorization (default: 10)
# ratingCol: column name for ratings (default: rating, current: rating)
# regParam: regularization parameter (>= 0). (default: 0.1, current: 0.01)
# seed: random seed. (default: 7537534487590478613)
# userCol: column name for user ids. Ids must be within the integer value range. (default: user, current: userId)

alsModel = als.fit(training)
predictions = alsModel.transform(test)

predictions.show()

# +------+-------+------+----------+----------+
# |userId|movieId|rating| timestamp|prediction|
# +------+-------+------+----------+----------+
# |     1|     13|   1.0|1424380312| 1.2091726|
# |     1|     19|   1.0|1424380312| 0.2089816|
# |     1|     36|   2.0|1424380312| 2.0141795|
# ...

##########################################################################################################

# 추천 수행 1: users 에게 items 를 추천함(buyer 관점)

res1 = alsModel.recommendForAllUsers(10)
# res1.show(truncate=False)
res1.selectExpr("userId", "explode(recommendations)").show()

# +------+---------------+
# |userId|            col|
# +------+---------------+
# |    20| {7, 4.2627573}|
# |    20| {75, 4.000416}|
# |    20|{94, 3.8061447}|
# |    20| {49, 3.742442}|
# |    20|{22, 3.6379368}|
# |    20|{64, 3.3643885}|
# |    20|{68, 3.3328223}|
# |    20| {77, 3.133869}|
# |    20|{51, 3.0384984}|
# |    20| {90, 2.923938}|
# |    10|  {49, 4.69336}|
# |    10|   {2, 3.94644}|
# |    10|{40, 3.7830536}|
# |    10|{32, 3.6905415}|
# |    10|{25, 3.2851338}|
# |    10|{87, 3.2453537}|
# |    10|{85, 3.1483278}|
# |    10| {62, 3.105364}|
# |    10|  {4, 2.991344}|
# |    10|{42, 2.9552217}|
# +------+---------------+
# only showing top 20 rows


# 추천 수행 2: items 를 users 에게 추천함(seller 관점)

res2 = alsModel.recommendForAllItems(10)
# res2.show(truncate=False)
res2.selectExpr("movieId", "explode(recommendations)").show()

# +-------+---------------+
# |movieId|            col|
# +-------+---------------+
# |     20| {7, 1.3987739}|
# |     20|{12, 1.2235224}|
# |     20|  {0, 1.220086}|
# |     20| {9, 1.1727351}|
# |     20|{14, 1.1386144}|
# |     20|{28, 1.0806793}|
# |     20|{18, 1.0610176}|
# |     20|{13, 1.0420345}|
# |     20|{26, 1.0284066}|
# |     20| {8, 1.0244251}|
# |     40|{19, 5.7484303}|
# |     40| {17, 5.101816}|
# |     40|{16, 4.2189054}|
# |     40|  {2, 3.925725}|
# |     40|{10, 3.7830536}|
# |     40|{23, 3.7816281}|
# |     40|{29, 2.9000337}|
# |     40| {4, 2.6789055}|
# |     40|{28, 2.3950284}|
# |     40|  {7, 2.393309}|
# +-------+---------------+
# only showing top 20 rows

##########################################################################################################

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator()\
  .setMetricName("rmse")\
  .setLabelCol("rating")\
  .setPredictionCol("prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = %f" % rmse)

# Root-mean-square error = 1.754253


from pyspark.mllib.evaluation import RegressionMetrics

regComparison = predictions.select("rating", "prediction")\
  .rdd.map(lambda x: (x(0), x(1)))
metrics = RegressionMetrics(regComparison)


from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics
from pyspark.sql.functions import col, expr

perUserActual = predictions\
  .where("rating > 2.5")\
  .groupBy("userId")\
  .agg(expr("collect_set(movieId) as movies"))

perUserPredictions = predictions\
  .orderBy(col("userId"), expr("prediction DESC"))\
  .groupBy("userId")\
  .agg(expr("collect_list(movieId) as movies"))

perUserActualvPred = perUserActual.join(perUserPredictions, ["userId"]).rdd\
  .map(lambda row: (row[1], row[2][:15]))
ranks = RankingMetrics(perUserActualvPred)

ranks.meanAveragePrecision    # 0.2643557831057831

ranks.precisionAt(5)    # 0.5416666666666665

