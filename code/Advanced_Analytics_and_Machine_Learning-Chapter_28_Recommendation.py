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

res1.show(truncate=False)

# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |userId|recommendations                                                                                                                                                          |
# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |20    |[{75, 4.567182}, {22, 4.413162}, {7, 4.0858417}, {94, 3.7357414}, {68, 3.6568038}, {87, 3.625748}, {51, 3.384547}, {90, 3.364666}, {69, 3.3539443}, {54, 3.226752}]      |
# |10    |[{39, 4.765701}, {2, 3.8162322}, {83, 3.7610846}, {29, 3.2957418}, {25, 3.2141266}, {93, 3.0254798}, {49, 3.0009253}, {42, 2.9334803}, {89, 2.828933}, {76, 2.7897449}]  |
# |0     |[{92, 3.8377638}, {9, 3.455524}, {1, 3.411512}, {63, 3.2490733}, {96, 3.2211843}, {52, 3.1901898}, {2, 3.0513813}, {91, 2.9160023}, {74, 2.9133668}, {28, 2.8170044}]    |
# |1     |[{1, 5.7067513}, {52, 4.2601013}, {31, 4.182467}, {29, 4.134969}, {68, 4.0132074}, {62, 3.772308}, {7, 3.7505543}, {95, 3.6210828}, {26, 3.4663694}, {8, 3.4018157}]     |
# |21    |[{53, 4.89849}, {90, 4.315899}, {74, 4.2555585}, {2, 3.874654}, {87, 3.8532224}, {96, 3.1015136}, {59, 2.9817662}, {43, 2.9563808}, {85, 2.717679}, {10, 2.5634325}]     |
# |11    |[{55, 5.67514}, {18, 5.0025907}, {69, 4.971443}, {30, 4.8674273}, {81, 4.446919}, {38, 4.032047}, {90, 4.02064}, {50, 3.797827}, {19, 3.7818596}, {13, 3.6982543}]       |
# |12    |[{27, 5.0791698}, {49, 4.9635506}, {17, 4.8688664}, {64, 4.6810794}, {35, 4.5474176}, {32, 4.122572}, {31, 3.9836535}, {50, 3.8506985}, {48, 3.5980184}, {81, 3.5830152}]|
# |22    |[{54, 5.1123924}, {88, 5.1122227}, {22, 5.0045943}, {75, 4.884646}, {51, 4.764607}, {85, 4.1530232}, {4, 3.9643128}, {69, 3.9563763}, {94, 3.9456716}, {32, 3.7761495}]  |
# |2     |[{39, 5.030879}, {8, 5.013279}, {83, 4.997412}, {34, 4.075298}, {92, 4.0003495}, {19, 3.903137}, {2, 3.8542337}, {74, 3.7208045}, {93, 3.704706}, {75, 3.498278}]        |
# |13    |[{93, 3.6638007}, {55, 3.2139325}, {83, 3.0992503}, {39, 3.0717378}, {29, 2.9982963}, {53, 2.949876}, {72, 2.832728}, {18, 2.7057805}, {74, 2.70071}, {87, 2.6394708}]   |
# |3     |[{51, 5.0833535}, {54, 4.8332534}, {69, 4.7630243}, {75, 4.7370214}, {22, 4.302495}, {88, 4.279877}, {18, 3.8508341}, {80, 3.821134}, {85, 3.5081344}, {4, 3.4328272}]   |
# |23    |[{32, 5.1050386}, {49, 5.0618663}, {55, 4.8558745}, {27, 4.8059196}, {48, 4.7922583}, {81, 4.769247}, {17, 4.6355495}, {35, 4.4944034}, {31, 4.480898}, {64, 4.4381227}] |
# |4     |[{76, 4.6642494}, {29, 3.9537332}, {93, 3.8785768}, {52, 3.8557537}, {41, 3.7121644}, {70, 3.6853933}, {72, 3.6700418}, {74, 3.537729}, {62, 3.3857453}, {2, 3.188877}]  |
# |24    |[{76, 5.569721}, {92, 5.2892866}, {52, 5.181234}, {90, 5.0559883}, {96, 5.028434}, {41, 4.784523}, {69, 4.6080227}, {74, 4.419853}, {63, 4.0237603}, {77, 3.9542065}]    |
# |14    |[{52, 5.0176253}, {76, 4.9413157}, {63, 4.711933}, {70, 4.474303}, {62, 4.0979733}, {72, 4.0459576}, {41, 4.045837}, {77, 3.9737492}, {96, 3.7313035}, {90, 3.655105}]   |
# |5     |[{87, 5.610059}, {53, 5.3441453}, {30, 4.9728203}, {55, 4.6655884}, {7, 4.313961}, {19, 4.0972037}, {74, 4.080526}, {75, 4.059453}, {34, 3.9159794}, {90, 3.8876305}]    |
# |15    |[{46, 4.7374606}, {1, 3.7166526}, {27, 3.4752715}, {64, 3.1470852}, {49, 3.0676396}, {26, 3.0151563}, {35, 2.8989573}, {98, 2.8674471}, {2, 2.861005}, {33, 2.7395725}]  |
# |25    |[{25, 5.1828837}, {71, 3.9794166}, {47, 3.9023068}, {2, 3.1961277}, {16, 3.0526738}, {63, 3.0381923}, {12, 2.891177}, {91, 2.8674934}, {34, 2.755371}, {1, 2.7144167}]   |
# |26    |[{22, 5.406607}, {94, 5.398534}, {75, 5.2810774}, {32, 5.1101174}, {7, 4.80592}, {88, 4.5661297}, {68, 4.35753}, {36, 4.1391783}, {51, 4.048987}, {73, 3.9190023}]       |
# |6     |[{25, 4.861323}, {2, 2.9050848}, {23, 2.8718088}, {67, 2.8657842}, {61, 2.7409368}, {35, 2.6197636}, {63, 2.5946488}, {47, 2.5121024}, {12, 2.4850962}, {91, 2.2783847}] |
# +------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# only showing top 20 rows

ratings.dtypes

# [('userId', 'int'),
#  ('recommendations', 'array<struct<movieId:int,rating:float>>')]

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

