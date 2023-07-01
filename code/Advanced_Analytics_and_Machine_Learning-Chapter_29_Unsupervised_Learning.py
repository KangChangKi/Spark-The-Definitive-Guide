from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test1").master("local").getOrCreate()

path_prefix = "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide"

from pyspark.ml.feature import VectorAssembler

va = VectorAssembler()\
  .setInputCols(["Quantity", "UnitPrice"])\
  .setOutputCol("features")

sales = va.transform(spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load(path_prefix + "/data/retail-data/by-day/*.csv")
  .limit(50)
  .coalesce(1)
  .where("Description IS NOT NULL"))

sales.cache()

sales.dtypes

# [('InvoiceNo', 'string'),
#  ('StockCode', 'string'),
#  ('Description', 'string'),
#  ('Quantity', 'int'),
#  ('InvoiceDate', 'timestamp'),
#  ('UnitPrice', 'double'),
#  ('CustomerID', 'double'),
#  ('Country', 'string'),
#  ('features', 'vector')]


#####################################################################################################

# k-means 는 가장 많이 사용되는 군집화 알고리즘 이다.

from pyspark.ml.clustering import KMeans

km = KMeans().setK(5)
print(km.explainParams())

# distanceMeasure: the distance measure. Supported options: 'euclidean' and 'cosine'. (default: euclidean)
# featuresCol: features column name. (default: features)
# initMode: The initialization algorithm. This can be either "random" to choose random points as initial cluster centers, or "k-means||" to use a parallel variant of k-means++ (default: k-means||)
# initSteps: The number of steps for k-means|| initialization mode. Must be > 0. (default: 2)
# k: The number of clusters to create. Must be > 1. (default: 2, current: 5)
# maxIter: max number of iterations (>= 0). (default: 20)
# predictionCol: prediction column name. (default: prediction)
# seed: random seed. (default: -4796463408658058907)
# tol: the convergence tolerance for iterative algorithms (>= 0). (default: 0.0001)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

kmModel = km.fit(sales)

summary = kmModel.summary
print(summary.clusterSizes) # number of points  ==> [5, 10, 20, 3, 12]
# kmModel.computeCost(sales)   # <== deprecated 되었다.

centers = kmModel.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Cluster Centers:
# [ 2.4  13.04]
# [23.2    0.956]
# [4.55   4.5965]
# [44.          1.16333333]
# [11.33333333  1.1       ]

#####################################################################################################

# BisectingKMeans = 이분법 k-means
# k-means vs. bisecting k-means = clusters 를 키워나감 vs. clusters 를 나눠나감 = 상향식 vs. 하향식
# 즉, bisecting k-means 는 초기에 모든 데이터가 하나의 cluster 를 형성하고, 그 이후에 k 개의 clusters 로 나눠나가게 된다.
# 
# bisecting k-means 는 k-means 보다 더 빠르다는 장점이 있다.

from pyspark.ml.clustering import BisectingKMeans

bkm = BisectingKMeans().setK(5).setMaxIter(5)
bkmModel = bkm.fit(sales)

summary = bkmModel.summary
print(summary.clusterSizes) # number of points  ==> [17, 10, 10, 10, 3]
# kmModel.computeCost(sales)

centers = kmModel.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# Cluster Centers:
# [ 2.4  13.04]
# [23.2    0.956]
# [4.55   4.5965]
# [44.          1.16333333]
# [11.33333333  1.1       ]

#####################################################################################################

# 다른 방식의 clustering 알고리즘 인데, 내가 이걸 쓸 일은 없을 것 같다.

from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture().setK(5)
print(gmm.explainParams())

# aggregationDepth: suggested depth for treeAggregate (>= 2). (default: 2)
# featuresCol: features column name. (default: features)
# k: Number of independent Gaussians in the mixture model. Must be > 1. (default: 2, current: 5)
# maxIter: max number of iterations (>= 0). (default: 100)
# predictionCol: prediction column name. (default: prediction)
# probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities. (default: probability)
# seed: random seed. (default: 2430035703214794489)
# tol: the convergence tolerance for iterative algorithms (>= 0). (default: 0.01)
# weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

model = gmm.fit(sales)

summary = model.summary
print(model.weights)   # [0.3573555483037232, 0.3381859417947245, 0.05274437737189763, 0.020000000000034268, 0.23171413252962048]

model.gaussiansDF.show(truncate=False)
summary.cluster.show()

# +---------------------------------------+--------------------------------------------------------------------------------------------------+
# |mean                                   |cov                                                                                               |
# +---------------------------------------+--------------------------------------------------------------------------------------------------+
# |[7.230442699317077,2.6924689910816992] |11.507225085267853  -5.457703069391315  \n-5.457703069391315  2.7927436648742843                  |
# |[18.47330847848463,1.1941583897130923] |50.02411307348583    -2.2022063137146217  \n-2.2022063137146217  0.2880757633314811               |
# |[45.096944413814064,1.2063414602551608]|26.4787715736666    1.0367498056772961  \n1.0367498056772961  0.2081168871673461                  |
# |[8.000000000017652,4.949999999998496]  |6.418190423562838E-10   -9.522693744640549E-11  \n-9.522693744640549E-11  3.633360279303048E-11   |
# |[2.9784014329620443,9.231012088705478] |1.1113667806643674  -2.547015552727621  \n-2.547015552727621  13.392635241665584                  |
# +---------------------------------------+--------------------------------------------------------------------------------------------------+
# 
# +----------+
# |prediction|
# +----------+
# |         2|
# |         1|
# |         1|
# ...

summary.clusterSizes   # [20, 17, 3, 1, 9]

summary.probability.show(truncate=False)

# +--------------------------------------------------------------------------------------------------------------+
# |probability                                                                                                   |
# +--------------------------------------------------------------------------------------------------------------+
# |[1.262903485073502E-13,1.3437676728190703E-7,0.9999998656228539,1.262903485073502E-13,1.262903485073502E-13]  |
# |[1.358429451667194E-14,0.9999999281428407,7.185711873516128E-8,1.3584294516627798E-14,1.3584294516627798E-14] |
# |[6.217941795863598E-14,0.9999979673994899,2.0326003236035316E-6,6.217941795863598E-14,6.217941795863598E-14]  |
# ...

#####################################################################################################

# 잠재 디리클레 할당(Latent Dirichlet Allocation) 은 텍스트 문서에 대한 토픽 모델링을 수행하는데 사용되는 계층적 군집화 모델이다.
# 텍스트 데이터를 LDA 에 입력하려면 먼저 CountVectorizer 를 사용해서 수치형으로 변환해야 한다.

# 데이터 준비.

from pyspark.ml.feature import Tokenizer, CountVectorizer

tkn = Tokenizer().setInputCol("Description").setOutputCol("DescOut")
tokenized = tkn.transform(sales.drop("features"))
cv = CountVectorizer()\
  .setInputCol("DescOut")\
  .setOutputCol("features")\
  .setVocabSize(500)\
  .setMinTF(0)\
  .setMinDF(0)\
  .setBinary(True)
cvFitted = cv.fit(tokenized)
prepped = cvFitted.transform(tokenized)

prepped.dtypes

# [('InvoiceNo', 'string'),
#  ('StockCode', 'string'),
#  ('Description', 'string'),
#  ('Quantity', 'int'),
#  ('InvoiceDate', 'timestamp'),
#  ('UnitPrice', 'double'),
#  ('CustomerID', 'double'),
#  ('Country', 'string'),
#  ('DescOut', 'array<string>'),
#  ('features', 'vector')]

from pyspark.ml.clustering import LDA

lda = LDA().setK(10).setMaxIter(5)
print(lda.explainParams())

# checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext. (default: 10)
# docConcentration: Concentration parameter (commonly named "alpha") for the prior placed on documents' distributions over topics ("theta"). (undefined)
# featuresCol: features column name. (default: features)
# k: The number of topics (clusters) to infer. Must be > 1. (default: 10, current: 10)
# keepLastCheckpoint: (For EM optimizer) If using checkpointing, this indicates whether to keep the last checkpoint. If false, then the checkpoint will be deleted. Deleting the checkpoint can cause failures if a data partition is lost, so set this bit with care. (default: True)
# learningDecay: Learning rate, set as anexponential decay rate. This should be between (0.5, 1.0] to guarantee asymptotic convergence. (default: 0.51)
# learningOffset: A (positive) learning parameter that downweights early iterations. Larger values make early iterations count less (default: 1024.0)
# maxIter: max number of iterations (>= 0). (default: 20, current: 5)
# optimizeDocConcentration: Indicates whether the docConcentration (Dirichlet parameter for document-topic distribution) will be optimized during training. (default: True)
# optimizer: Optimizer or inference algorithm used to estimate the LDA model.  Supported: online, em (default: online)
# seed: random seed. (default: 4222874917626949620)
# subsamplingRate: Fraction of the corpus to be sampled and used in each iteration of mini-batch gradient descent, in range (0, 1]. (default: 0.05)
# topicConcentration: Concentration parameter (commonly named "beta" or "eta") for the prior placed on topic' distributions over terms. (undefined)
# topicDistributionCol: Output column with estimates of the topic mixture distribution for each document (often called "theta" in the literature). Returns a vector of zeros for an empty document. (default: topicDistribution)

model = lda.fit(prepped)

model.describeTopics(3).show(truncate=False)

# +-----+-------------+------------------------------------------------------------------+
# |topic|termIndices  |termWeights                                                       |
# +-----+-------------+------------------------------------------------------------------+
# |0    |[100, 54, 48]|[0.011448150041538053, 0.011028837060604438, 0.010735977958264023]|
# |1    |[128, 28, 5] |[0.011575303554002736, 0.011375483124511847, 0.01079122619531574] |
# |2    |[56, 17, 8]  |[0.00889813548235879, 0.00867973077785975, 0.00847787545202918]   |
# |3    |[0, 1, 3]    |[0.018391532846619346, 0.0179044368008798, 0.01773164331742865]   |
# |4    |[12, 2, 76]  |[0.016552340796312272, 0.01450988952307516, 0.014428811958965258] |
# |5    |[57, 87, 108]|[0.011795295039908773, 0.011091074338826726, 0.010971764073673462]|
# |6    |[25, 13, 105]|[0.008724329020375385, 0.008718746386075869, 0.008476584875151594]|
# |7    |[19, 93, 98] |[0.00952534743551824, 0.009497726916550256, 0.0087768807040834]   |
# |8    |[67, 13, 43] |[0.011744904173206882, 0.011671630840113169, 0.011163529916579784]|
# |9    |[11, 56, 70] |[0.008871864619504552, 0.008762942740563367, 0.008756548421378211]|
# +-----+-------------+------------------------------------------------------------------+

cvFitted.vocabulary

# ['water',
#  'hot',
#  'vintage',
#  'bottle',
#  'paperweight',
#  '6',
#  'home',
#  ...

