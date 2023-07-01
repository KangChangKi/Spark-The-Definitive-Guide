from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test1").master("local").getOrCreate()

path_prefix = "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide"

sales = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load(path_prefix + "/data/retail-data/by-day/*.csv")\
  .coalesce(5)\
  .where("Description IS NOT NULL")
fakeIntDF = spark.read.parquet(path_prefix + "/data/simple-ml-integers")
simpleDF = spark.read.json(path_prefix + "/data/simple-ml")
scaleDF = spark.read.parquet(path_prefix + "/data/simple-ml-scaling")

sales.cache()
sales.show(truncate=False)

from pyspark.ml.feature import Tokenizer

tkn = Tokenizer().setInputCol("Description").setOutputCol("res")

tkn.transform(sales.select("Description")).show(truncate=False)


#################################################################################################################

# COMMAND ----------

from pyspark.ml.feature import RFormula

supervised = RFormula(formula="lab ~ . + color:value1 + color:value2")
supervised.fit(simpleDF).transform(simpleDF).show()


# COMMAND ----------

from pyspark.ml.feature import SQLTransformer

basicTransformation = SQLTransformer()\
  .setStatement("""
    SELECT sum(Quantity), count(*), CustomerID
    FROM __THIS__
    GROUP BY CustomerID
  """)

basicTransformation.transform(sales).show()


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

va = VectorAssembler().setInputCols(["int1", "int2", "int3"]).setOutputCol("res")
va.transform(fakeIntDF).show()


# COMMAND ----------

contDF = spark.range(20).selectExpr("cast(id as double)")


# COMMAND ----------

from pyspark.ml.feature import Bucketizer

bucketBorders = [-1.0, 5.0, 10.0, 250.0, 600.0]
bucketer = Bucketizer().setSplits(bucketBorders).setInputCol("id").setOutputCol("res")
bucketer.transform(contDF).show()


# COMMAND ----------

from pyspark.ml.feature import QuantileDiscretizer

bucketer = QuantileDiscretizer().setNumBuckets(5).setInputCol("id").setOutputCol("res")
fittedBucketer = bucketer.fit(contDF)
fittedBucketer.transform(contDF).show()


# COMMAND ----------

from pyspark.ml.feature import StandardScaler

sScaler = StandardScaler().setInputCol("features").setOutputCol("res")
sScaler.fit(scaleDF).transform(scaleDF).show(truncate=False)


# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler

minMax = MinMaxScaler().setMin(5).setMax(10).setInputCol("features").setOutputCol("res")
fittedminMax = minMax.fit(scaleDF)
fittedminMax.transform(scaleDF).show()


# COMMAND ----------

from pyspark.ml.feature import MaxAbsScaler

maScaler = MaxAbsScaler().setInputCol("features").setOutputCol("res")
fittedmaScaler = maScaler.fit(scaleDF)
fittedmaScaler.transform(scaleDF).show(truncate=False)


# COMMAND ----------

from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors

scaleUpVec = Vectors.dense(10.0, 15.0, 20.0)
scalingUp = ElementwiseProduct()\
  .setScalingVec(scaleUpVec)\
  .setInputCol("features")\
  .setOutputCol("res")
scalingUp.transform(scaleDF).show()


# COMMAND ----------

from pyspark.ml.feature import Normalizer

manhattanDistance = Normalizer().setP(1).setInputCol("features").setOutputCol("res")
manhattanDistance.transform(scaleDF).show(truncate=False)


# COMMAND ----------

from pyspark.ml.feature import StringIndexer

lblIndxr = StringIndexer().setInputCol("lab").setOutputCol("labelInd")
idxRes = lblIndxr.fit(simpleDF).transform(simpleDF)
idxRes.show()


# COMMAND ----------

valIndexer = StringIndexer().setInputCol("value1").setOutputCol("valueInd")
valIndexer.fit(simpleDF).transform(simpleDF).show()


# COMMAND ----------

from pyspark.ml.feature import IndexToString

labelReverse = IndexToString().setInputCol("labelInd").setOutputCol("res")
labelReverse.transform(idxRes).show()


# COMMAND ----------

from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vectors

idxIn = spark.createDataFrame([
  (Vectors.dense(1, 2, 3),1),
  (Vectors.dense(2, 5, 6),2),
  (Vectors.dense(1, 8, 9),3)
]).toDF("features", "label")
indxr = VectorIndexer()\
  .setInputCol("features")\
  .setOutputCol("idxed")\
  .setMaxCategories(2)
indxr.fit(idxIn).transform(idxIn).show()


# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

lblIndxr = StringIndexer().setInputCol("color").setOutputCol("colorInd")
colorLab = lblIndxr.fit(simpleDF).transform(simpleDF.select("color"))
ohe = OneHotEncoder().setInputCol("colorInd").setOutputCol("res")
ohe.fit(colorLab).transform(colorLab).show()


#################################################################################################################

# 단어를 공백 1개로 잘라서 tokenize 를 수행한다.
# 공백이 2개면 빈 공백 1개가 token 이 된다.

from pyspark.ml.feature import Tokenizer

tkn = Tokenizer().setInputCol("Description").setOutputCol("DescOut")
tokenized = tkn.transform(sales.select("Description"))
tokenized.show(20, False)


# Java 의 regexp 를 사용할 수 있다.
# 공백이 2개 이상일 경우를 다음과 같이 처리할 수 있다.

from pyspark.ml.feature import RegexTokenizer

rt = RegexTokenizer()\
  .setInputCol("Description")\
  .setOutputCol("DescOut")\
  .setPattern("\s+")\
  .setToLowercase(True)
rt.transform(sales.select("Description")).show(20, False)


# `.setGaps(False)` 를 하면 pattern 이 매칭하는 단어들만 리턴된다.
# 여기서 "gaps" = "delimiters" 라서 pattern 이 delimiter 를 의미하는지(True, default), word 를 의미하는지(False) 를 선택하는 옵션이다.
# 
# ES 에 stopwords 와 keepwords 가 있는데, 서로 반대의 개념이다.
# 여기서 `.setGaps(True)` 이면 stopwords 로 동작하고,
# `.setGaps(False)` 이면 keepwords 로 동작하고, 

from pyspark.ml.feature import RegexTokenizer

rt = RegexTokenizer()\
  .setInputCol("Description")\
  .setOutputCol("DescOut")\
  .setPattern("\w+")\
  .setGaps(False)\
  .setToLowercase(True)
rt.transform(sales.select("Description")).show(20, False)


# stopwords 를 제거한다.
# 아쉽게도 korean 은 지원하지 않는다.
# 그러나 그냥 list of word 를 쓰면 된다.

from pyspark.ml.feature import StopWordsRemover

# englishStopWords = StopWordsRemover.loadDefaultStopWords("english")
englishStopWords = ["a", "an", "the", "rabbit"]
stops = StopWordsRemover()\
  .setStopWords(englishStopWords)\
  .setInputCol("DescOut")
stops.transform(tokenized).show(truncate=False)


# ngram 을 만들 수 있다.

from pyspark.ml.feature import NGram

unigram = NGram().setInputCol("DescOut").setN(1).setOutputCol("res_1")
bigram = NGram().setInputCol("DescOut").setN(2).setOutputCol("res_2")
unigram.transform(tokenized.select("DescOut")).show(truncate=False)
bigram.transform(tokenized.select("DescOut")).show(truncate=False)


# CountVectorizer 는 단어 vector 를 숫자 vector 로 변환시킨다.
# 그냥 vector 의 elements 를 단어 -> 숫자로 바꾸는 것이다.
# 개념적으로는 Bag of Words(BOW) 를 나타낸다.

from pyspark.ml.feature import CountVectorizer

cv = CountVectorizer()\
  .setInputCol("DescOut")\
  .setOutputCol("countVec")\
  .setVocabSize(500)\
  .setMinTF(1)\
  .setMinDF(2)
fittedCV = cv.fit(tokenized)
fittedCV.transform(tokenized).show(truncate=False)

# +-----------------------------------+------------------------------------------+---------------------------------------------------+
# |Description                        |DescOut                                   |countVec                                           |
# +-----------------------------------+------------------------------------------+---------------------------------------------------+
# |RABBIT NIGHT LIGHT                 |[rabbit, night, light]                    |(500,[149,185,212],[1.0,1.0,1.0])                  |
# |DOUGHNUT LIP GLOSS                 |[doughnut, lip, gloss]                    |(500,[462,463,492],[1.0,1.0,1.0])                  |
# |12 MESSAGE CARDS WITH ENVELOPES    |[12, message, cards, with, envelopes]     |(500,[35,41,166],[1.0,1.0,1.0])                    |
# ...
# 
# countVec 의 형태를 보면, 3개의 원소를 가지는 tuple 이다.
# sparse vector 이고, (총 어휘 크기, 어휘에 포함된 단어 색인, 특정 단어의 출현 빈도) 를 의미한다.

tfIdfIn = tokenized\
  .where("array_contains(DescOut, 'red')")\
  .select("DescOut")\
  .limit(10)
tfIdfIn.show(n=10, truncate=False)

# +---------------------------------------+
# |DescOut                                |
# +---------------------------------------+
# |[gingham, heart, , doorstop, red]      |
# |[red, floral, feltcraft, shoulder, bag]|
# |[alarm, clock, bakelike, red]          |
# ...
# 
# 위와 같이 DataFrame 의 다른 연산과 조합이 가능하다.


# HashingTF 가 이전의 CountVectorizer 와 다른 점은 hashing 과정을 포함한다는 것이다.
# 그러면 얻을 수 있는 장점은 단어 -> hashing -> 숫자 로 변환을 하기 때문에, 숫자 -> 단어 로 역추적이 불가능해진다.
#
# usecase 1:
# 그래서 HashingTF 는 비밀을 다룰 때 좋다.
# 반면, 데이터가 비밀이 아니라면 CountVectorizer 를 쓰는게 더 효율적이라서 좋다.
#
# usecase 2:
# HashingTF 와 IDF 를 조합해서 단어별 TF 값과, IDF 값을 얻을 수 있다.
# 반면, CountVectorizer 는 IDF 와 조합할 수 없다.

from pyspark.ml.feature import HashingTF, IDF

tf = HashingTF()\
  .setInputCol("DescOut")\
  .setOutputCol("TFOut")\
  .setNumFeatures(10000)
idf = IDF()\
  .setInputCol("TFOut")\
  .setOutputCol("IDFOut")\
  .setMinDocFreq(2)

idf.fit(tf.transform(tfIdfIn)).transform(tf.transform(tfIdfIn)).show(n=10, truncate=False)

# +---------------------------------------+-----------------------------------------------------+------------------------------------------------------------
# |DescOut                                |TFOut                                                |IDFOut
# +---------------------------------------+-----------------------------------------------------+------------------------------------------------------------
# |[alarm, clock, bakelike, red]          |(10000,[52,4995,8737,9001],[1.0,1.0,1.0,1.0])        |(10000,[52,4995,8737,9001],[0.0,0.0,0.0,0.0])                                                                    |
# |[pin, cushion, babushka, red]          |(10000,[52,610,2490,7153],[1.0,1.0,1.0,1.0])         |(10000,[52,610,2490,7153],[0.0,0.0,0.0,1.2992829841302609])                                                      |
# |[red, retrospot, mini, cases]          |(10000,[52,547,6703,8448],[1.0,1.0,1.0,1.0])         |(10000,[52,547,6703,8448],[0.0,0.0,0.0,1.0116009116784799])
# ...
# 
# TF 값과 IDF 값을 얻을 수 있음에 주의해라.
# CountVectorizer 는 TF 값만 얻을 수 있다.


# 학습 및 사용이 쉽고, 엔티티 인식, 모호성 제거, 구문 분석, 태그 지정 및 기계 번역을 포함한 여러 가지 자연어 처리 애플리케이션에서 유용하게 사용된다.
# 토큰 형태이면서 연속적이고 자유형의 텍스트에서 가장 잘 동작한다.

from pyspark.ml.feature import Word2Vec

# Input data: Each row is a bag of words from a sentence or document.
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text",
  outputCol="result")
model = word2Vec.fit(documentDF)
result = model.transform(documentDF)
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

# Text: [Hi, I, heard, about, Spark] =>
# Vector: [-0.08150363978929819,-0.019268991798162462,-0.06999367889366113]
# 
# Text: [I, wish, Java, could, use, case, classes] =>
# Vector: [-0.004938643159610884,-0.033724620938301086,-0.01584516681448024]
# 
# Text: [Logistic, regression, models, are, neat] =>
# Vector: [0.003283527493476868,-0.06138550490140915,-0.010659397765994073]

#################################################################################################################

# 대표적인 차원 축소 테크닉이다.
# feature selection 도 포함하고 있다.
# 유일한 단점은 feature 값이 변경되어서 해석이 불가능해지고, 원복도 안된다는 점이 있다.
# 또한 k (축소할 차원 값) 을 설정하는게 무척 중요한데, 공식적인 가이드가 없다.

from pyspark.ml.feature import PCA

pca = PCA().setInputCol("features").setK(2).setOutputCol("res")
pca.fit(scaleDF).transform(scaleDF).show(n=20, truncate=False)

# +---+--------------+------------------------------------------+
# |id |features      |res                                       |
# +---+--------------+------------------------------------------+
# |0  |[1.0,0.1,-1.0]|[0.0713719499248417,-0.4526654888147805]  |
# |1  |[2.0,1.1,1.0] |[-1.6804946984073723,1.2593401322219198]  |
# |0  |[1.0,0.1,-1.0]|[0.0713719499248417,-0.4526654888147805]  |
# ...
# 
# PCA 로 차원 축소를 하면 features 는 의미를 알 수 없는 값들이 되어 버린다.


# RFormula 는 domain 지식을 활용할 수 있게 만든다.
# 즉, domain 지식을 반영한 interactions 을 추가할 수 있고, 어떤 terms 는 삭제할 수 도 있다
# 
# 그런데, interactions 를 추가하는 것도 일이다.
# PolynomialExpansion 는 cartesian product 방식으로 모든 가능한 interactions 를 추가한다.
# degree = 차수 = 몇개의 변수를 조합할지를 결정한다.
# 예) 변수가 3 개 이고, 차수가 2 이면 => 3**2 = 9 개의 조합이 된다.
#    a, b, c => a*a + a*b + a*c + b*a + b*b + b*c + c*a + c*b + c*c (9개의 조합)
# 예) 변수가 3 개 이고, 차수가 3 이면 => 3**3 = 27 개의 조합이 된다.
#    a, b, c => a*a*a + a*a*b + a*a*c + a*b*a + a*b*b + a*b*c + a*c*a + a*c*b + a*c*c + ... (27개의 조합)
# 
# 위의 확장된 결과를 보면, `a*a*a` 는 `a` 와 같고, `a*a*b + a*b*a + b*a*a + a*b*b + b*b*a + ...` 는 `a:b` 와 같음을 알 수 있다.
# 즉, 모든 terms 를 다 커버하고 있다.
# 
# 그래서 domain 지식 없이 모든 interactions 를 포함시킬 때 좋다.
# 반면, 과적합(overfit) 을 초례할 수도 있는 단점이 있다.

from pyspark.ml.feature import PolynomialExpansion

pe = PolynomialExpansion().setInputCol("features").setDegree(2).setOutputCol("res")
pe.transform(scaleDF).show(truncate=False)

# +---+--------------+-----------------------------------------------------------------------------------+
# |id |features      |res                                                                                |
# +---+--------------+-----------------------------------------------------------------------------------+
# |0  |[1.0,0.1,-1.0]|[1.0,1.0,0.1,0.1,0.010000000000000002,-1.0,-1.0,-0.1,1.0]                          |
# |1  |[2.0,1.1,1.0] |[2.0,4.0,1.1,2.2,1.2100000000000002,1.0,2.0,1.1,1.0]                               |
# |0  |[1.0,0.1,-1.0]|[1.0,1.0,0.1,0.1,0.010000000000000002,-1.0,-1.0,-0.1,1.0]                          |
# ...
# 마치 cartesian product 처럼 원소들이 곱해져 있는 것을 볼 수 있다.


# 통계적 검정을 사용한 feature selection 방법이다.
# 반면, PCA 는 vector space 를 사용한 feature selection 방법이다.
# 둘다 feature selection 을 수행하기 때문에 k (number of features) 를 지정하는게 똑같다.
# 즉, 둘다 같은 목적을 가지기 때문에 입력받는 parameter 가 비슷하다.

from pyspark.ml.feature import ChiSqSelector, Tokenizer

tkn = Tokenizer().setInputCol("Description").setOutputCol("DescOut")
tokenized = tkn\
  .transform(sales.select("Description", "CustomerId"))\
  .where("CustomerId IS NOT NULL")
prechi = fittedCV.transform(tokenized)\
  .where("CustomerId IS NOT NULL")

chisq = ChiSqSelector()\
  .setFeaturesCol("countVec")\
  .setLabelCol("CustomerId")\
  .setNumTopFeatures(2)\
  .setOutputCol("res")

chisq.fit(prechi).transform(prechi)\
  .drop("customerId", "Description", "DescOut").show(truncate=False)

# +---------------------------------------------------+-------------------+
# |countVec                                           |res                |
# +---------------------------------------------------+-------------------+
# |(500,[149,185,212],[1.0,1.0,1.0])                  |(2,[],[])          |
# ...
# |(500,[0,1,49,70,365,366],[1.0,1.0,1.0,1.0,1.0,1.0])|(2,[0,1],[1.0,1.0])|
# ...
# |(500,[0,2,6,328,405],[1.0,1.0,1.0,1.0,1.0])        |(2,[0],[1.0])      |
# ...
# 
# `.setNumTopFeatures(2)` 로 설정했기 때문에 res 에 2 개의 원소들만 나온다.
# 유의미한 features 가 최대 2개 만 선택된 것을 볼 수 있다.
# 예)
# (2,[0,1],[1.0,1.0])  <== features 가 2개 선택됨.
# (2,[0],[1.0])       <== features 가 1개 선택됨.
# (2,[],[])           <== features 가 0개 선택됨.

#################################################################################################################

# 변환자 저장하기

fittedPCA = pca.fit(scaleDF)
fittedPCA.write().overwrite().save(path_prefix + "/tmp/fittedPCA")


# 변환자 불러오기

from pyspark.ml.feature import PCAModel

loadedPCA = PCAModel.load(path_prefix + "/tmp/fittedPCA")
loadedPCA.transform(scaleDF).show(truncate=False)
