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


# SQLTransformer 를 사용할 때 UDF 를 등록해서 사용할 수 있다.
# 이러면 custom transformer 를 만들어서 사용할 필요가 없다.
# SQLTransformer + UDF 조합으로 custom code 로 transforming 수행할 수 있기 때문이다.

from pyspark.ml.feature import SQLTransformer

from pyspark.sql.functions import udf
from pyspark.sql.types import *

@udf(returnType=IntegerType())
def add2(a, b):
    return a + b

spark.udf.register("add2", add2)

basicTransformation = SQLTransformer()\
  .setStatement("""
    SELECT sum(Quantity), add2(sum(Quantity), 99), count(*), CustomerID
    FROM __THIS__
    GROUP BY CustomerID
  """)

basicTransformation.transform(sales).show()

# +-------------+-----------------------+--------+----------+
# |sum(Quantity)|add2(sum(Quantity), 99)|count(1)|CustomerID|
# +-------------+-----------------------+--------+----------+
# |          119|                    218|      62|   14452.0|
# |          440|                    539|     143|   16916.0|
# |          630|                    729|      72|   17633.0|
# |           34|                    133|       6|   14768.0|
# |         1542|                   1641|      30|   13094.0|
# |          854|                    953|     117|   17884.0|
# |           97|                    196|      12|   16596.0|
# |          290|                    389|      98|   13607.0|
# |          541|                    640|      27|   14285.0|
# |          244|                    343|      31|   16561.0|
# |          756|                    855|      67|   15145.0|
# |           83|                    182|      13|   16858.0|
# |           56|                    155|       4|   13160.0|
# |         8873|                   8972|      80|   16656.0|
# |          241|                    340|      43|   16212.0|
# |          258|                    357|      23|   13142.0|
# |           67|                    166|      14|   13811.0|
# |          569|                    668|      57|   12550.0|
# |           84|                    183|       4|   15160.0|
# |          954|                   1053|      82|   15067.0|
# +-------------+-----------------------+--------+----------+
# only showing top 20 rows

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

# +-----------------------------------+------------------------------------------+
# |Description                        |DescOut                                   |
# +-----------------------------------+------------------------------------------+
# |RABBIT NIGHT LIGHT                 |[rabbit, night, light]                    |
# |DOUGHNUT LIP GLOSS                 |[doughnut, lip, gloss]                    |
# |12 MESSAGE CARDS WITH ENVELOPES    |[12, message, cards, with, envelopes]     |
# |BLUE HARMONICA IN BOX              |[blue, harmonica, in, box]                |
# |GUMBALL COAT RACK                  |[gumball, coat, rack]                     |
# |SKULLS  WATER TRANSFER TATTOOS     |[skulls, , water, transfer, tattoos]      |
# |FELTCRAFT GIRL AMELIE KIT          |[feltcraft, girl, amelie, kit]            |
# |CAMOUFLAGE LED TORCH               |[camouflage, led, torch]                  |
# |WHITE SKULL HOT WATER BOTTLE       |[white, skull, hot, water, bottle]        |
# |ENGLISH ROSE HOT WATER BOTTLE      |[english, rose, hot, water, bottle]       |
# |HOT WATER BOTTLE KEEP CALM         |[hot, water, bottle, keep, calm]          |
# |SCOTTIE DOG HOT WATER BOTTLE       |[scottie, dog, hot, water, bottle]        |
# |ROSE CARAVAN DOORSTOP              |[rose, caravan, doorstop]                 |
# |GINGHAM HEART  DOORSTOP RED        |[gingham, heart, , doorstop, red]         |
# |STORAGE TIN VINTAGE LEAF           |[storage, tin, vintage, leaf]             |
# |SET OF 4 KNICK KNACK TINS POPPIES  |[set, of, 4, knick, knack, tins, poppies] |
# |POPCORN HOLDER                     |[popcorn, holder]                         |
# |GROW A FLYTRAP OR SUNFLOWER IN TIN |[grow, a, flytrap, or, sunflower, in, tin]|
# |AIRLINE BAG VINTAGE WORLD CHAMPION |[airline, bag, vintage, world, champion]  |
# |AIRLINE BAG VINTAGE JET SET BROWN  |[airline, bag, vintage, jet, set, brown]  |
# +-----------------------------------+------------------------------------------+
# only showing top 20 rows

# Java 의 regexp 를 사용할 수 있다.
# 공백이 2개 이상일 경우를 다음과 같이 처리할 수 있다.

from pyspark.ml.feature import RegexTokenizer

rt = (RegexTokenizer()
  .setInputCol("Description")
  .setOutputCol("DescOut")
  .setPattern("\s+")
  .setToLowercase(True))
rt.transform(sales.select("Description")).show(20, False)

# +-----------------------------------+------------------------------------------+
# |Description                        |DescOut                                   |
# +-----------------------------------+------------------------------------------+
# |RABBIT NIGHT LIGHT                 |[rabbit, night, light]                    |
# |DOUGHNUT LIP GLOSS                 |[doughnut, lip, gloss]                    |
# |12 MESSAGE CARDS WITH ENVELOPES    |[12, message, cards, with, envelopes]     |
# |BLUE HARMONICA IN BOX              |[blue, harmonica, in, box]                |
# |GUMBALL COAT RACK                  |[gumball, coat, rack]                     |
# |SKULLS  WATER TRANSFER TATTOOS     |[skulls, water, transfer, tattoos]        |
# |FELTCRAFT GIRL AMELIE KIT          |[feltcraft, girl, amelie, kit]            |
# |CAMOUFLAGE LED TORCH               |[camouflage, led, torch]                  |
# |WHITE SKULL HOT WATER BOTTLE       |[white, skull, hot, water, bottle]        |
# |ENGLISH ROSE HOT WATER BOTTLE      |[english, rose, hot, water, bottle]       |
# |HOT WATER BOTTLE KEEP CALM         |[hot, water, bottle, keep, calm]          |
# |SCOTTIE DOG HOT WATER BOTTLE       |[scottie, dog, hot, water, bottle]        |
# |ROSE CARAVAN DOORSTOP              |[rose, caravan, doorstop]                 |
# |GINGHAM HEART  DOORSTOP RED        |[gingham, heart, doorstop, red]           |
# |STORAGE TIN VINTAGE LEAF           |[storage, tin, vintage, leaf]             |
# |SET OF 4 KNICK KNACK TINS POPPIES  |[set, of, 4, knick, knack, tins, poppies] |
# |POPCORN HOLDER                     |[popcorn, holder]                         |
# |GROW A FLYTRAP OR SUNFLOWER IN TIN |[grow, a, flytrap, or, sunflower, in, tin]|
# |AIRLINE BAG VINTAGE WORLD CHAMPION |[airline, bag, vintage, world, champion]  |
# |AIRLINE BAG VINTAGE JET SET BROWN  |[airline, bag, vintage, jet, set, brown]  |
# +-----------------------------------+------------------------------------------+
# only showing top 20 rows

# `.setGaps(False)` 를 하면 pattern 이 매칭하는 단어들만 리턴된다.
# 여기서 "gaps" = "delimiters" 라서 pattern 이 delimiter 를 의미하는지(True, default), word 를 의미하는지(False) 를 선택하는 옵션이다.
# 
# ES 에 stopwords 와 keepwords 가 있는데, 서로 반대의 개념이다.
# 여기서 `.setGaps(True)` 이면 regexp pattern 이 stopwords 로 동작하고,
# `.setGaps(False)` 이면 regexp pattern 이 keepwords 로 동작하고, 

from pyspark.ml.feature import RegexTokenizer

rt = (RegexTokenizer()
  .setInputCol("Description")
  .setOutputCol("DescOut")
  .setPattern("\w+")
  .setGaps(False)   # <-- 여기
  .setToLowercase(True))
rt.transform(sales.select("Description")).show(20, False)

# +-----------------------------------+------------------------------------------+
# |Description                        |DescOut                                   |
# +-----------------------------------+------------------------------------------+
# |RABBIT NIGHT LIGHT                 |[rabbit, night, light]                    |
# |DOUGHNUT LIP GLOSS                 |[doughnut, lip, gloss]                    |
# |12 MESSAGE CARDS WITH ENVELOPES    |[12, message, cards, with, envelopes]     |
# |BLUE HARMONICA IN BOX              |[blue, harmonica, in, box]                |
# |GUMBALL COAT RACK                  |[gumball, coat, rack]                     |
# |SKULLS  WATER TRANSFER TATTOOS     |[skulls, water, transfer, tattoos]        |
# |FELTCRAFT GIRL AMELIE KIT          |[feltcraft, girl, amelie, kit]            |
# |CAMOUFLAGE LED TORCH               |[camouflage, led, torch]                  |
# |WHITE SKULL HOT WATER BOTTLE       |[white, skull, hot, water, bottle]        |
# |ENGLISH ROSE HOT WATER BOTTLE      |[english, rose, hot, water, bottle]       |
# |HOT WATER BOTTLE KEEP CALM         |[hot, water, bottle, keep, calm]          |
# |SCOTTIE DOG HOT WATER BOTTLE       |[scottie, dog, hot, water, bottle]        |
# |ROSE CARAVAN DOORSTOP              |[rose, caravan, doorstop]                 |
# |GINGHAM HEART  DOORSTOP RED        |[gingham, heart, doorstop, red]           |
# |STORAGE TIN VINTAGE LEAF           |[storage, tin, vintage, leaf]             |
# |SET OF 4 KNICK KNACK TINS POPPIES  |[set, of, 4, knick, knack, tins, poppies] |
# |POPCORN HOLDER                     |[popcorn, holder]                         |
# |GROW A FLYTRAP OR SUNFLOWER IN TIN |[grow, a, flytrap, or, sunflower, in, tin]|
# |AIRLINE BAG VINTAGE WORLD CHAMPION |[airline, bag, vintage, world, champion]  |
# |AIRLINE BAG VINTAGE JET SET BROWN  |[airline, bag, vintage, jet, set, brown]  |
# +-----------------------------------+------------------------------------------+
# only showing top 20 rows

# stopwords 를 제거한다.
# 아쉽게도 korean 은 지원하지 않는다.
# 그러나 그냥 list of word 를 쓰면 된다.

from pyspark.ml.feature import StopWordsRemover

# englishStopWords = StopWordsRemover.loadDefaultStopWords("english")
englishStopWords = ["a", "an", "the", "rabbit"]
stops = (StopWordsRemover()
  .setStopWords(englishStopWords)
  # .setStopWords(["rabbit", "night", "light"])   # <-- 이렇게 custom stopwords 를 설정할 수 있다.
  .setInputCol("DescOut"))
stops.transform(tokenized).show(truncate=False)

# +-----------------------------------+------------------------------------------+-----------------------------------------+
# |Description                        |DescOut                                   |StopWordsRemover_fa96c037009e__output    |
# +-----------------------------------+------------------------------------------+-----------------------------------------+
# |RABBIT NIGHT LIGHT                 |[rabbit, night, light]                    |[night, light]                           |
# |DOUGHNUT LIP GLOSS                 |[doughnut, lip, gloss]                    |[doughnut, lip, gloss]                   |
# |12 MESSAGE CARDS WITH ENVELOPES    |[12, message, cards, with, envelopes]     |[12, message, cards, with, envelopes]    |
# |BLUE HARMONICA IN BOX              |[blue, harmonica, in, box]                |[blue, harmonica, in, box]               |
# |GUMBALL COAT RACK                  |[gumball, coat, rack]                     |[gumball, coat, rack]                    |
# |SKULLS  WATER TRANSFER TATTOOS     |[skulls, , water, transfer, tattoos]      |[skulls, , water, transfer, tattoos]     |
# |FELTCRAFT GIRL AMELIE KIT          |[feltcraft, girl, amelie, kit]            |[feltcraft, girl, amelie, kit]           |
# |CAMOUFLAGE LED TORCH               |[camouflage, led, torch]                  |[camouflage, led, torch]                 |
# |WHITE SKULL HOT WATER BOTTLE       |[white, skull, hot, water, bottle]        |[white, skull, hot, water, bottle]       |
# |ENGLISH ROSE HOT WATER BOTTLE      |[english, rose, hot, water, bottle]       |[english, rose, hot, water, bottle]      |
# |HOT WATER BOTTLE KEEP CALM         |[hot, water, bottle, keep, calm]          |[hot, water, bottle, keep, calm]         |
# |SCOTTIE DOG HOT WATER BOTTLE       |[scottie, dog, hot, water, bottle]        |[scottie, dog, hot, water, bottle]       |
# |ROSE CARAVAN DOORSTOP              |[rose, caravan, doorstop]                 |[rose, caravan, doorstop]                |
# |GINGHAM HEART  DOORSTOP RED        |[gingham, heart, , doorstop, red]         |[gingham, heart, , doorstop, red]        |
# |STORAGE TIN VINTAGE LEAF           |[storage, tin, vintage, leaf]             |[storage, tin, vintage, leaf]            |
# |SET OF 4 KNICK KNACK TINS POPPIES  |[set, of, 4, knick, knack, tins, poppies] |[set, of, 4, knick, knack, tins, poppies]|
# |POPCORN HOLDER                     |[popcorn, holder]                         |[popcorn, holder]                        |
# |GROW A FLYTRAP OR SUNFLOWER IN TIN |[grow, a, flytrap, or, sunflower, in, tin]|[grow, flytrap, or, sunflower, in, tin]  |
# |AIRLINE BAG VINTAGE WORLD CHAMPION |[airline, bag, vintage, world, champion]  |[airline, bag, vintage, world, champion] |
# |AIRLINE BAG VINTAGE JET SET BROWN  |[airline, bag, vintage, jet, set, brown]  |[airline, bag, vintage, jet, set, brown] |
# +-----------------------------------+------------------------------------------+-----------------------------------------+
# only showing top 20 rows

# ngram 을 만들 수 있다.

from pyspark.ml.feature import NGram

unigram = NGram().setInputCol("DescOut").setN(1).setOutputCol("res_1")
bigram = NGram().setInputCol("DescOut").setN(2).setOutputCol("res_2")
unigram.transform(tokenized.select("DescOut")).show(truncate=False)
bigram.transform(tokenized.select("DescOut")).show(truncate=False)

# +------------------------------------------+------------------------------------------+
# |DescOut                                   |res_1                                     |
# +------------------------------------------+------------------------------------------+
# |[rabbit, night, light]                    |[rabbit, night, light]                    |
# |[doughnut, lip, gloss]                    |[doughnut, lip, gloss]                    |
# |[12, message, cards, with, envelopes]     |[12, message, cards, with, envelopes]     |
# |[blue, harmonica, in, box]                |[blue, harmonica, in, box]                |
# |[gumball, coat, rack]                     |[gumball, coat, rack]                     |
# |[skulls, , water, transfer, tattoos]      |[skulls, , water, transfer, tattoos]      |
# |[feltcraft, girl, amelie, kit]            |[feltcraft, girl, amelie, kit]            |
# |[camouflage, led, torch]                  |[camouflage, led, torch]                  |
# |[white, skull, hot, water, bottle]        |[white, skull, hot, water, bottle]        |
# |[english, rose, hot, water, bottle]       |[english, rose, hot, water, bottle]       |
# |[hot, water, bottle, keep, calm]          |[hot, water, bottle, keep, calm]          |
# |[scottie, dog, hot, water, bottle]        |[scottie, dog, hot, water, bottle]        |
# |[rose, caravan, doorstop]                 |[rose, caravan, doorstop]                 |
# |[gingham, heart, , doorstop, red]         |[gingham, heart, , doorstop, red]         |
# |[storage, tin, vintage, leaf]             |[storage, tin, vintage, leaf]             |
# |[set, of, 4, knick, knack, tins, poppies] |[set, of, 4, knick, knack, tins, poppies] |
# |[popcorn, holder]                         |[popcorn, holder]                         |
# |[grow, a, flytrap, or, sunflower, in, tin]|[grow, a, flytrap, or, sunflower, in, tin]|
# |[airline, bag, vintage, world, champion]  |[airline, bag, vintage, world, champion]  |
# |[airline, bag, vintage, jet, set, brown]  |[airline, bag, vintage, jet, set, brown]  |
# +------------------------------------------+------------------------------------------+
# only showing top 20 rows
# 
# +------------------------------------------+-------------------------------------------------------------------+
# |DescOut                                   |res_2                                                              |
# +------------------------------------------+-------------------------------------------------------------------+
# |[rabbit, night, light]                    |[rabbit night, night light]                                        |
# |[doughnut, lip, gloss]                    |[doughnut lip, lip gloss]                                          |
# |[12, message, cards, with, envelopes]     |[12 message, message cards, cards with, with envelopes]            |
# |[blue, harmonica, in, box]                |[blue harmonica, harmonica in, in box]                             |
# |[gumball, coat, rack]                     |[gumball coat, coat rack]                                          |
# |[skulls, , water, transfer, tattoos]      |[skulls ,  water, water transfer, transfer tattoos]                |
# |[feltcraft, girl, amelie, kit]            |[feltcraft girl, girl amelie, amelie kit]                          |
# |[camouflage, led, torch]                  |[camouflage led, led torch]                                        |
# |[white, skull, hot, water, bottle]        |[white skull, skull hot, hot water, water bottle]                  |
# |[english, rose, hot, water, bottle]       |[english rose, rose hot, hot water, water bottle]                  |
# |[hot, water, bottle, keep, calm]          |[hot water, water bottle, bottle keep, keep calm]                  |
# |[scottie, dog, hot, water, bottle]        |[scottie dog, dog hot, hot water, water bottle]                    |
# |[rose, caravan, doorstop]                 |[rose caravan, caravan doorstop]                                   |
# |[gingham, heart, , doorstop, red]         |[gingham heart, heart ,  doorstop, doorstop red]                   |
# |[storage, tin, vintage, leaf]             |[storage tin, tin vintage, vintage leaf]                           |
# |[set, of, 4, knick, knack, tins, poppies] |[set of, of 4, 4 knick, knick knack, knack tins, tins poppies]     |
# |[popcorn, holder]                         |[popcorn holder]                                                   |
# |[grow, a, flytrap, or, sunflower, in, tin]|[grow a, a flytrap, flytrap or, or sunflower, sunflower in, in tin]|
# |[airline, bag, vintage, world, champion]  |[airline bag, bag vintage, vintage world, world champion]          |
# |[airline, bag, vintage, jet, set, brown]  |[airline bag, bag vintage, vintage jet, jet set, set brown]        |
# +------------------------------------------+-------------------------------------------------------------------+
# only showing top 20 rows

# ==> 근데 unigram 은 의미가 없고 bigram 부터 의미가 있다.

# CountVectorizer 는 단어 vector 를 숫자 vector 로 변환시킨다.
# 그냥 vector 의 elements 를 단어 -> 숫자로 바꾸는 것이다.
# 개념적으로는 Bag of Words(BOW) 를 나타낸다.
# 
# 단어의 count = TF 이다.

from pyspark.ml.feature import CountVectorizer

cv = (CountVectorizer()
  .setInputCol("DescOut")
  .setOutputCol("countVec")
  .setVocabSize(500)
  .setMinTF(1)
  .setMinDF(2))
fittedCV = cv.fit(tokenized)
fittedCV.transform(tokenized).show(truncate=False)

# +-----------------------------------+------------------------------------------+---------------------------------------------------+
# |Description                        |DescOut                                   |countVec                                           |
# +-----------------------------------+------------------------------------------+---------------------------------------------------+
# |RABBIT NIGHT LIGHT                 |[rabbit, night, light]                    |(500,[150,185,212],[1.0,1.0,1.0])                  |
# |DOUGHNUT LIP GLOSS                 |[doughnut, lip, gloss]                    |(500,[462,463,492],[1.0,1.0,1.0])                  |
# |12 MESSAGE CARDS WITH ENVELOPES    |[12, message, cards, with, envelopes]     |(500,[35,41,166],[1.0,1.0,1.0])                    |
# ...
# 
# countVec 의 형태를 보면, 3개의 원소를 가지는 tuple 이다.
# sparse vector 이고, (총 어휘 크기, 어휘에 포함된 단어 색인, 특정 단어의 출현 빈도) 를 의미한다.

assert set([fittedCV.vocabulary[e] for e in [150,185,212]]) == set(['light', 'rabbit', 'night'])
assert fittedCV.getVocabSize() == 500

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

tfIdfIn = (tokenized
  .where("array_contains(DescOut, 'red')")
  .select("DescOut")
  .limit(10))
tfIdfIn.show(n=10, truncate=False)

# +---------------------------------------+
# |DescOut                                |
# +---------------------------------------+
# |[gingham, heart, , doorstop, red]      |
# |[red, floral, feltcraft, shoulder, bag]|
# |[alarm, clock, bakelike, red]          |
# |[pin, cushion, babushka, red]          |
# |[red, retrospot, mini, cases]          |
# |[red, kitchen, scales]                 |
# |[gingham, heart, , doorstop, red]      |
# |[large, red, babushka, notebook]       |
# |[red, retrospot, oven, glove]          |
# |[red, retrospot, plate]                |
# +---------------------------------------+
# 
# 'red' 에 대한 IDF 를 구하기 위해서 위와 같이 데이터를 만들었다.

tf = (HashingTF()
  .setInputCol("DescOut")
  .setOutputCol("TFOut")
  .setNumFeatures(10000))
idf = (IDF()
  .setInputCol("TFOut")   # <-- 이전 tf 의 결과 column 을 연결 한다.
  .setOutputCol("IDFOut")
  .setMinDocFreq(2))

res = tf.transform(tfIdfIn)
idf.fit(res).transform(res).show(n=10, truncate=False)   # <-- IDF 는 전체를 미리 scan 해야 구할 수 있기 때문에 fit 과 transform 연이어 해야 한다.

# +---------------------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
# |DescOut                                |TFOut                                                |IDFOut                                                                                                           |
# +---------------------------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
# |[gingham, heart, , doorstop, red]      |(10000,[52,804,3372,6594,9808],[1.0,1.0,1.0,1.0,1.0])|(10000,[52,804,3372,6594,9808],[0.0,1.2992829841302609,1.2992829841302609,1.2992829841302609,1.2992829841302609])|
# |[red, floral, feltcraft, shoulder, bag]|(10000,[50,52,415,6756,8005],[1.0,1.0,1.0,1.0,1.0])  |(10000,[50,52,415,6756,8005],[0.0,0.0,0.0,0.0,0.0])                                                              |
# |[alarm, clock, bakelike, red]          |(10000,[52,4995,8737,9001],[1.0,1.0,1.0,1.0])        |(10000,[52,4995,8737,9001],[0.0,0.0,0.0,0.0])                                                                    |
# ...
# 
# 위 방식으로 TF 값과 IDF 값을 모두 얻을 수 있음에 주의해라.
# CountVectorizer 는 TF 값만 얻을 수 있다.


# word2vec 은 학습 및 사용이 쉽고, 엔티티 인식, 모호성 제거, 구문 분석, 태그 지정 및 기계 번역을 포함한 여러 가지 자연어 처리 애플리케이션에서 유용하게 사용된다.
# 토큰 형태이면서 연속적이고 자유형의 텍스트에서 가장 잘 동작한다.

from pyspark.ml.feature import Word2Vec

# Input data: Each row is a bag of words from a sentence or document.
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)
result = model.transform(documentDF)
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

# 이름이 "word"2vec 인데, 실제 동작은 "sentence"2vec 임에 주의해라.

# Text: [Hi, I, heard, about, Spark] =>
# Vector: [0.0017009951174259187,0.049393982067704206,0.05914585739374161]
# 
# Text: [I, wish, Java, could, use, case, classes] =>
# Vector: [-0.021818850255970444,-0.06204076483845711,-0.07141843544585363]
# 
# Text: [Logistic, regression, models, are, neat] =>
# Vector: [-0.08085853531956673,0.03675719648599625,-0.036668000370264055]

#################################################################################################################

# 대표적인 차원 축소 테크닉이다.
# feature selection 도 포함하고 있다.
# 유일한 단점은 feature 값이 변경되어서 해석이 불가능해지고, 원복도 안된다는 점이 있다.
# 또한 k (축소 결과의 차원 값) 을 설정하는게 무척 중요한데, 공식적인 가이드가 없다.

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
# PCA 로 차원 축소를 하면 column "res" 는 의미를 알 수 없는 값들이 되어 버린다.
# 왜냐하면 차원 축들이 변경되었기 때문에 그 안의 vectors 도 따라서 위치값이 바뀌었기 때문이다.

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
tokenized = (tkn
  .transform(sales.select("Description", "CustomerId"))
  .where("CustomerId IS NOT NULL"))
prechi = (fittedCV.transform(tokenized)
  .where("CustomerId IS NOT NULL"))

chisq = (ChiSqSelector()
  .setFeaturesCol("countVec")
  .setLabelCol("CustomerId")
  .setNumTopFeatures(2)
  .setOutputCol("res"))

(chisq.fit(prechi).transform(prechi)
  .drop("customerId", "Description", "DescOut").show(truncate=False))

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

from pyspark.ml.feature import PCAModel           # <-- 동일한 class 명에 "Model" 만 붙이면 된다.

loadedPCA = PCAModel.load(path_prefix + "/tmp/fittedPCA")
loadedPCA.transform(scaleDF).show(truncate=False)
