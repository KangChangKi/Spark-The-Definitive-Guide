from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test1").master("local").getOrCreate()

path_prefix = "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide"

df = spark.read.load(path_prefix + "/data/regression")

df.dtypes    # [('features', 'vector'), ('label', 'double')]
df.show()

# +--------------+-----+
# |      features|label|
# +--------------+-----+
# |[3.0,10.1,3.0]|  2.0|
# | [2.0,1.1,1.0]|  1.0|
# |[1.0,0.1,-1.0]|  0.0|
# |[1.0,0.1,-1.0]|  0.0|
# | [2.0,4.1,1.0]|  2.0|
# +--------------+-----+

##################################################################################################

# 선형 회기 분석은 입력 특징들의 선형 조합(가중치가 곱해진 각 특징을 합한 값, sum of product)이 가우시안 오차(Gaussian error)와 함게 최종 결과로 산출된다고 가정한다.
# "선형" = "linear" 는 선형 조합(linear combination) 에서 유래한다.
# ==> LinearRegression 은 선형 조합 + 가우시안 에러 이다.

from pyspark.ml.regression import LinearRegression

lr = LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
print(lr.explainParams())

# aggregationDepth: suggested depth for treeAggregate (>= 2). (default: 2)
# elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. (default: 0.0, current: 0.8)
# epsilon: The shape parameter to control the amount of robustness. Must be > 1.0. Only valid when loss is huber (default: 1.35)
# featuresCol: features column name. (default: features)
# fitIntercept: whether to fit an intercept term. (default: True)
# labelCol: label column name. (default: label)
# loss: The loss function to be optimized. Supported options: squaredError, huber. (default: squaredError)
# maxBlockSizeInMB: maximum memory in MB for stacking input data into blocks. Data is stacked within partitions. If more than remaining data size in a partition then it is adjusted to the data size. Default 0.0 represents choosing optimal value, depends on specific algorithm. Must be >= 0. (default: 0.0)
# maxIter: max number of iterations (>= 0). (default: 100, current: 10)
# predictionCol: prediction column name. (default: prediction)
# regParam: regularization parameter (>= 0). (default: 0.0, current: 0.3)
# solver: The solver algorithm for optimization. Supported options: auto, normal, l-bfgs. (default: auto)
# standardization: whether to standardize the training features before fitting the model. (default: True)
# tol: the convergence tolerance for iterative algorithms (>= 0). (default: 1e-06)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

lrModel = lr.fit(df)

summary = lrModel.summary
summary.residuals.show()

print(summary.totalIterations)
print(summary.objectiveHistory)
print(summary.rootMeanSquaredError)
print(summary.r2)

# +--------------------+
# |           residuals|
# +--------------------+
# | 0.12805046585610147|
# |-0.14468269261572053|
# |-0.41903832622420595|
# |-0.41903832622420595|
# |  0.8547088792080308|
# +--------------------+
# 
# 5
# [0.5000000000000001, 0.4315295810362787, 0.3132335933881022, 0.31225692666554117, 0.309150608198303, 0.30915058933480255]
# 0.47308424392175996
# 0.7202391226912209

# 선형 회귀의 성능 metric 은 RMSE(root mean squared error) 과 R2 이다.
#
# RMSE 는 0 에 가까울 수록 더 정확한 예측 성능을 낸다고 해석하면 된다. (label 과 prediction 의 차이가 적다)
# R2 는 1에 가까울 수록 회귀 모델이 데이터를 더 잘 설명한다고 할 수 있다. (변수가 데이터를 잘 설명한다, 전체 변수와 데이터의 연관도 라고 보면 된다)

# R2 를 그림으로 보면 더 잘 이해할 수 있다.
# scatter chart 가 있고, 그 안에 line 이 있다.
# data points 가 line 과 유사할 수록 R2 는 1 에 가깝게 된다.
# 반면, data points 가 random 해서 원형을 이루면 R2 는 0 에 가깝게 된다.
# 즉, 변수가 얼마나 데이터를 잘 설명하는지를 의미하는게 R2 이다.

# RMSE vs. R2
# RMSE 는 model 의 정확도를 의미하자면,
# R2 는 features 와 데이터의 연관도를 의미한다.
# 그래서 RMSE 는 model 의 성능 metric 이고, R2 는 model 의 features 의 selection 에 관한 metric 이다.
# 
# 그래서 RMSE 가 0 에 가까우면 "model 의 성능이 좋다" 라고 하고,
# R2 가 1 에 가까우면 "model 이 잘 만들어졌다" 또는 "model 의 feature-selection 이 잘 되었다" 라고 한다.
# 
# RMSE 가 1 에 가까운데, R2 를 봤더니 0 에 가깝다면 데이터를 더 잘 묘사하는 신규 features 를 발굴해야 한다.
# RMSE 가 1 에 가까운데, R2 를 봤더니 1 에 가깝다면 features 는 좋은데 학습이 제대로 되지 않은 것이라서 iteration 등을 높이는 등의 처치를 해야 한다.


# 이전에 다뤘던 LinearRegression 은 선형 조합 + 가우시안 에러 이다.
# 그런데 에러 부분은 다른 확률분포로 교체할 수 있게 generalization 한게 GeneralizedLinearRegression 이다.
# family 로 여러 확률분포들 중 하나를 선택하여 적용할 수 있다.

from pyspark.ml.regression import GeneralizedLinearRegression

glr = GeneralizedLinearRegression()\
  .setFamily("gaussian")\
  .setLink("identity")\
  .setMaxIter(10)\
  .setRegParam(0.3)\
  .setLinkPredictionCol("linkOut")
print(glr.explainParams())

# aggregationDepth: suggested depth for treeAggregate (>= 2). (default: 2)
# family: The name of family which is a description of the error distribution to be used in the model. Supported options: gaussian (default), binomial, poisson, gamma and tweedie. (default: gaussian, current: gaussian)
# featuresCol: features column name. (default: features)
# fitIntercept: whether to fit an intercept term. (default: True)
# labelCol: label column name. (default: label)
# link: The name of link function which provides the relationship between the linear predictor and the mean of the distribution function. Supported options: identity, log, inverse, logit, probit, cloglog and sqrt. (current: identity)
# linkPower: The index in the power link function. Only applicable to the Tweedie family. (undefined)
# linkPredictionCol: link prediction (linear predictor) column name (current: linkOut)
# maxIter: max number of iterations (>= 0). (default: 25, current: 10)
# offsetCol: The offset column name. If this is not set or empty, we treat all instance offsets as 0.0 (undefined)
# predictionCol: prediction column name. (default: prediction)
# regParam: regularization parameter (>= 0). (default: 0.0, current: 0.3)
# solver: The solver algorithm for optimization. Supported options: irls. (default: irls)
# tol: the convergence tolerance for iterative algorithms (>= 0). (default: 1e-06)
# variancePower: The power in the variance function of the Tweedie distribution which characterizes the relationship between the variance and mean of the distribution. Only applicable for the Tweedie family. Supported values: 0 and [1, Inf). (default: 0.0)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

glrModel = glr.fit(df)

##################################################################################################

# 예전에 다뤘던 DecisionTreeClassifier 와 DecisionTreeRegressor 와 다른 것은 prediction 과 label 의 data type 이 다르다는 것 뿐이다.
# 즉, classifier 는 categorical 이고, regressor 는 numeric 이다.

from pyspark.ml.regression import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
print(dtr.explainParams())

# cacheNodeIds: If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval. (default: False)
# checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext. (default: 10)
# featuresCol: features column name. (default: features)
# impurity: Criterion used for information gain calculation (case-insensitive). Supported options: variance (default: variance)
# labelCol: label column name. (default: label)
# leafCol: Leaf indices column name. Predicted leaf index of each instance in each tree by preorder. (default: )
# maxBins: Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature. (default: 32)
# maxDepth: Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30]. (default: 5)
# maxMemoryInMB: Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size. (default: 256)
# minInfoGain: Minimum information gain for a split to be considered at a tree node. (default: 0.0)
# minInstancesPerNode: Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1. (default: 1)
# minWeightFractionPerNode: Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5). (default: 0.0)
# predictionCol: prediction column name. (default: prediction)
# seed: random seed. (default: -3044224918519422125)
# varianceCol: column name for the biased sample variance of prediction. (undefined)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

dtrModel = dtr.fit(df)


# 예전에 classifiers 에서 봤던 것 처럼, RandomForestRegressor 는 수평 확장이고, GBTRegressor 는 수직 확장이다.

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor

rf =  RandomForestRegressor()
print(rf.explainParams())

# bootstrap: Whether bootstrap samples are used when building trees. (default: True)
# cacheNodeIds: If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval. (default: False)
# checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext. (default: 10)
# featureSubsetStrategy: The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'. If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features), 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)), 'n' (when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features), use n features). default = 'auto' (default: auto)
# featuresCol: features column name. (default: features)
# impurity: Criterion used for information gain calculation (case-insensitive). Supported options: variance (default: variance)
# labelCol: label column name. (default: label)
# leafCol: Leaf indices column name. Predicted leaf index of each instance in each tree by preorder. (default: )
# maxBins: Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature. (default: 32)
# maxDepth: Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30]. (default: 5)
# maxMemoryInMB: Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size. (default: 256)
# minInfoGain: Minimum information gain for a split to be considered at a tree node. (default: 0.0)
# minInstancesPerNode: Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1. (default: 1)
# minWeightFractionPerNode: Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5). (default: 0.0)
# numTrees: Number of trees to train (>= 1). (default: 20)
# predictionCol: prediction column name. (default: prediction)
# seed: random seed. (default: 7842565446311787658)
# subsamplingRate: Fraction of the training data used for learning each decision tree, in range (0, 1]. (default: 1.0)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

rfModel = rf.fit(df)


gbt = GBTRegressor()
print(gbt.explainParams())

# cacheNodeIds: If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval. (default: False)
# checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext. (default: 10)
# featureSubsetStrategy: The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'. If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features), 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)), 'n' (when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features), use n features). default = 'auto' (default: all)
# featuresCol: features column name. (default: features)
# impurity: Criterion used for information gain calculation (case-insensitive). Supported options: variance (default: variance)
# labelCol: label column name. (default: label)
# leafCol: Leaf indices column name. Predicted leaf index of each instance in each tree by preorder. (default: )
# lossType: Loss function which GBT tries to minimize (case-insensitive). Supported options: squared, absolute (default: squared)
# maxBins: Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature. (default: 32)
# maxDepth: Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30]. (default: 5)
# maxIter: max number of iterations (>= 0). (default: 20)
# maxMemoryInMB: Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size. (default: 256)
# minInfoGain: Minimum information gain for a split to be considered at a tree node. (default: 0.0)
# minInstancesPerNode: Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1. (default: 1)
# minWeightFractionPerNode: Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5). (default: 0.0)
# predictionCol: prediction column name. (default: prediction)
# seed: random seed. (default: 4394017947289811279)
# stepSize: Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator. (default: 0.1)
# subsamplingRate: Fraction of the training data used for learning each decision tree, in range (0, 1]. (default: 1.0)
# validationIndicatorCol: name of the column that indicates whether each row is for training or for validation. False indicates training; true indicates validation. (undefined)
# validationTol: Threshold for stopping early when fit with validation is used. If the error rate on the validation input changes by less than the validationTol, then learning will stop early (before `maxIter`). This parameter is ignored when fit without validation is used. (default: 0.01)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

gbtModel = gbt.fit(df)

##################################################################################################

# fit 수행.

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

glr = GeneralizedLinearRegression().setFamily("gaussian").setLink("identity")
pipeline = Pipeline().setStages([glr])
params = ParamGridBuilder().addGrid(glr.regParam, [0, 0.5, 1]).build()
evaluator = RegressionEvaluator()\
  .setMetricName("rmse")\
  .setPredictionCol("prediction")\
  .setLabelCol("label")
cv = CrossValidator()\
  .setEstimator(pipeline)\
  .setEvaluator(evaluator)\
  .setEstimatorParamMaps(params)\
  .setNumFolds(2) # should always be 3 or more but this dataset is small
model = cv.fit(df)


# transform 수행.

from pyspark.mllib.evaluation import RegressionMetrics

out = model.transform(df)\
  .select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = RegressionMetrics(out)
print("MSE: " + str(metrics.meanSquaredError))
print("RMSE: " + str(metrics.rootMeanSquaredError))
print("R-squared: " + str(metrics.r2))
print("MAE: " + str(metrics.meanAbsoluteError))
print("Explained variance: " + str(metrics.explainedVariance))

# MSE: 0.15705521472392633
# RMSE: 0.3963019236944559
# R-squared: 0.803680981595092
# MAE: 0.3141104294478528
# Explained variance: 0.6429447852760738
