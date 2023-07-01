from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test1").master("local").getOrCreate()

path_prefix = "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide"

bInput = spark.read.format("parquet").load(path_prefix + "/data/binary-classification")\
  .selectExpr("features", "cast(label as double) as label")

bInput.show(truncate=False)

# +--------------+-----+
# |features      |label|
# +--------------+-----+
# |[3.0,10.1,3.0]|1.0  |
# |[1.0,0.1,-1.0]|0.0  |
# |[1.0,0.1,-1.0]|0.0  |
# |[2.0,1.1,1.0] |1.0  |
# |[2.0,1.1,1.0] |1.0  |
# +--------------+-----+

###############################################################################################

# logistic regression 은 가장 많이 사용되는 분류 모델이다.

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression()
print(lr.explainParams())  # see all parameters

# aggregationDepth: suggested depth for treeAggregate (>= 2). (default: 2)
# elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. (default: 0.0)
# family: The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial (default: auto)
# featuresCol: features column name. (default: features)
# fitIntercept: whether to fit an intercept term. (default: True)
# labelCol: label column name. (default: label)
# lowerBoundsOnCoefficients: The lower bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the shape (1, number of features) for binomial regression, or (number of classes, number of features) for multinomial regression. (undefined)
# lowerBoundsOnIntercepts: The lower bounds on intercepts if fitting under bound constrained optimization. The bounds vector size must beequal with 1 for binomial regression, or the number oflasses for multinomial regression. (undefined)
# maxBlockSizeInMB: maximum memory in MB for stacking input data into blocks. Data is stacked within partitions. If more than remaining data size in a partition then it is adjusted to the data size. Default 0.0 represents choosing optimal value, depends on specific algorithm. Must be >= 0. (default: 0.0)
# maxIter: max number of iterations (>= 0). (default: 100)
# predictionCol: prediction column name. (default: prediction)
# probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities. (default: probability)
# rawPredictionCol: raw prediction (a.k.a. confidence) column name. (default: rawPrediction)
# regParam: regularization parameter (>= 0). (default: 0.0)
# standardization: whether to standardize the training features before fitting the model. (default: True)
# threshold: Threshold in binary classification prediction, in range [0, 1]. If threshold and thresholds are both set, they must match.e.g. if threshold is p, then thresholds must be equal to [1-p, p]. (default: 0.5)
# thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0, excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold. (undefined)
# tol: the convergence tolerance for iterative algorithms (>= 0). (default: 1e-06)
# upperBoundsOnCoefficients: The upper bounds on coefficients if fitting under bound constrained optimization. The bound matrix must be compatible with the shape (1, number of features) for binomial regression, or (number of classes, number of features) for multinomial regression. (undefined)
# upperBoundsOnIntercepts: The upper bounds on intercepts if fitting under bound constrained optimization. The bound vector size must be equal with 1 for binomial regression, or the number of classes for multinomial regression. (undefined)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)
# 
# 설명이 간결하면서도 잘 나온다.

lrModel = lr.fit(bInput)


# LR model 의 coefficients 와 intercept 를 알 수 있다.

print(lrModel.coefficients)    # [18.722385741661263,-0.569368855734081,9.361192870830655]
print(lrModel.intercept)      # -28.043295118689397


# LR model 의 요약을 볼 수 있다.

summary = lrModel.summary
print(summary.areaUnderROC)
summary.roc.show()
summary.pr.show()

# 1.0

# +---+------------------+
# |FPR|               TPR|
# +---+------------------+
# |0.0|               0.0|
# |0.0|0.3333333333333333|
# |0.0|               1.0|
# |1.0|               1.0|
# |1.0|               1.0|
# +---+------------------+

# +------------------+---------+
# |            recall|precision|
# +------------------+---------+
# |               0.0|      1.0|
# |0.3333333333333333|      1.0|
# |               1.0|      1.0|
# |               1.0|      0.6|
# +------------------+---------+


# objectiveHistory 는 PySpark ML의 모델 훈련 과정에서 각 반복(iteration)마다 발생한 목적 함수(objective function)의 값을 기록하는 속성 이다.
# 이것을 통해서 반복 횟수가 충분했는지 아니면 다른 파라미터를 조정할 필요가 있는지 확인할 수 있다.

summary.objectiveHistory

# [0.6730116670092565,
#  0.30533476678669746,
#  0.19572951692227342,
#  0.0823856071750674,
#  0.039904390712412544,
#  0.01918760572997783,
#  0.009480513129879561,
#  0.00470079397539896,
#  0.0023428240050888376,
#  0.0011692212872630402,
#  0.0005841333526454082,
#  0.0002919384368144791,
#  0.00014593757317783441,
#  7.295887614374728e-05,
#  3.647309882223473e-05,
#  1.8228017083425185e-05,
#  9.09575546492761e-06,
#  4.505306292845877e-06,
#  2.1743484095164223e-06,
#  1.0422594942126728e-06,
#  5.280808738948732e-07,
#  2.628531186444674e-07,
#  1.316603223969379e-07,
#  6.578498712561231e-08,
#  3.290121373800987e-08,
#  1.6448921648782615e-08,
#  8.224786126081331e-09]

###############################################################################################

# 의사결정트리는 가장 친근한 모델인데, 인간의 의사결정과 유사하기 때문이다.
# 그래서 사람에게 모델을 설명하여 이해시키기 좋다는 장점이 있다.

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier()
print(dt.explainParams())

# cacheNodeIds: If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval. (default: False)
# checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext. (default: 10)
# featuresCol: features column name. (default: features)
# impurity: Criterion used for information gain calculation (case-insensitive). Supported options: entropy, gini (default: gini)
# labelCol: label column name. (default: label)
# leafCol: Leaf indices column name. Predicted leaf index of each instance in each tree by preorder. (default: )
# maxBins: Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature. (default: 32)
# maxDepth: Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30]. (default: 5)
# maxMemoryInMB: Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size. (default: 256)
# minInfoGain: Minimum information gain for a split to be considered at a tree node. (default: 0.0)
# minInstancesPerNode: Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1. (default: 1)
# minWeightFractionPerNode: Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5). (default: 0.0)
# predictionCol: prediction column name. (default: prediction)
# probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities. (default: probability)
# rawPredictionCol: raw prediction (a.k.a. confidence) column name. (default: rawPrediction)
# seed: random seed. (default: -3007514338314502155)
# thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0, excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold. (undefined)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

dtModel = dt.fit(bInput)

###############################################################################################

# random forest 와 gradient boost tree 는 decision tree 로부터 확장된 알고리즘들이다.
# 
# random forest 는 "tree" -> "forest" 처럼 도메인 전문가 trees 를 모아서 집단지성을 발휘하도록 확장한 것이고,
# 전문가들이 서로 견제를 하여 과적합(overfit) 을 방지하는 장점이 있다.
# 
# gradient boost tree 는 "tree" 가 그대로 유지된다.
# 왜냐하면 tree 를 수직으로 쌓아서 하나의 tree 모습을 유지하기 때문이다.
# 즉, pipeline 과 같이 동작한다.
# 
# random forest 와 gradient boost tree 의 차이점은,
# random forest 는 도메인 전문가들의 의견을 공평하게 mean 하여 합치는데,
# gradient boost tree 는 도메인 전문가들의 의견을 차등하게 weight 를 주어서 합친다.
# 
# 그래서 random forest 와 gradient boost tree 의 차이점을 형태로 보면,
# random forest 는 옆으로 퍼진 형태이고,
# gradient boost tree 는 위아래로 퍼진 형태이다.

from pyspark.ml.classification import RandomForestClassifier

rfClassifier = RandomForestClassifier()
print(rfClassifier.explainParams())
trainedModel1 = rfClassifier.fit(bInput)


from pyspark.ml.classification import GBTClassifier

gbtClassifier = GBTClassifier()
print(gbtClassifier.explainParams())

# cacheNodeIds: If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval. (default: False)
# checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext. (default: 10)
# featureSubsetStrategy: The number of features to consider for splits at each tree node. Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'. If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression), 'all' (use all features), 'onethird' (use 1/3 of the features), 'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)), 'n' (when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features), use n features). default = 'auto' (default: all)
# featuresCol: features column name. (default: features)
# impurity: Criterion used for information gain calculation (case-insensitive). Supported options: variance (default: variance)
# labelCol: label column name. (default: label)
# leafCol: Leaf indices column name. Predicted leaf index of each instance in each tree by preorder. (default: )
# lossType: Loss function which GBT tries to minimize (case-insensitive). Supported options: logistic (default: logistic)
# maxBins: Max number of bins for discretizing continuous features.  Must be >=2 and >= number of categories for any categorical feature. (default: 32)
# maxDepth: Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. Must be in range [0, 30]. (default: 5)
# maxIter: max number of iterations (>= 0). (default: 20)
# maxMemoryInMB: Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size. (default: 256)
# minInfoGain: Minimum information gain for a split to be considered at a tree node. (default: 0.0)
# minInstancesPerNode: Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1. (default: 1)
# minWeightFractionPerNode: Minimum fraction of the weighted sample count that each child must have after split. If a split causes the fraction of the total weight in the left or right child to be less than minWeightFractionPerNode, the split will be discarded as invalid. Should be in interval [0.0, 0.5). (default: 0.0)
# predictionCol: prediction column name. (default: prediction)
# probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities. (default: probability)
# rawPredictionCol: raw prediction (a.k.a. confidence) column name. (default: rawPrediction)
# seed: random seed. (default: -1110379546510933616)
# stepSize: Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator. (default: 0.1)
# subsamplingRate: Fraction of the training data used for learning each decision tree, in range (0, 1]. (default: 1.0)
# thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0, excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold. (undefined)
# validationIndicatorCol: name of the column that indicates whether each row is for training or for validation. False indicates training; true indicates validation. (undefined)
# validationTol: Threshold for stopping early when fit with validation is used. If the error rate on the validation input changes by less than the validationTol, then learning will stop early (before `maxIter`). This parameter is ignored when fit without validation is used. (default: 0.01)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

trainedModel2 = gbtClassifier.fit(bInput)


###############################################################################################

# 베이즈 정리에 기반한 알고리즘이다.
# 이 모델의 핵심 가정은 데이터의 모든 특징이 서로 독립적이라는 것이다.
# 그래서 "naive" 라는 말이 붙었다.
# 모든 변수들간에 interaction 이 없다는 가정은 "naive" 하기 때문이다.
# 
# naive bayes classifier 는 모든 input features 가 음수가 아니어야 한다는 제한이 있다.

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()
print(nb.explainParams())

# featuresCol: features column name. (default: features)
# labelCol: label column name. (default: label)
# modelType: The model type which is a string (case-sensitive). Supported options: multinomial (default), bernoulli and gaussian. (default: multinomial)
# predictionCol: prediction column name. (default: prediction)
# probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities. (default: probability)
# rawPredictionCol: raw prediction (a.k.a. confidence) column name. (default: rawPrediction)
# smoothing: The smoothing parameter, should be >= 0, default is 1.0 (default: 1.0)
# thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0, excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold. (undefined)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

trainedModel3 = nb.fit(bInput.where("label != 0"))


###############################################################################################

from pyspark.mllib.evaluation import BinaryClassificationMetrics

def evaluate(model):
  """
  여러 models 를 실행하고 평가하기 위해서 function 을 만듦.
  """
  out = model.transform(bInput)\
    .select("prediction", "label")\
    .rdd.map(lambda x: (float(x[0]), float(x[1])))
  metrics = BinaryClassificationMetrics(out)

  print(metrics.areaUnderPR)
  print(metrics.areaUnderROC)

  try:
    print("Receiver Operating Characteristic")
    metrics.roc.toDF().show()
  except:
    pass

  try:
    summary = model.summary
    print(summary.areaUnderROC)
    summary.roc.show()
    summary.pr.show()
  except:
    pass

evaluate(lrModel)
evaluate(dtModel)
evaluate(trainedModel1)
evaluate(trainedModel2)
evaluate(trainedModel3)

