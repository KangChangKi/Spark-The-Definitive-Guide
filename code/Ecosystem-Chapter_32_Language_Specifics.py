from pyspark.sql import SparkSession

spark = (SparkSession.builder.appName("test1").master("local")
  .getOrCreate())

path_prefix = "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide"

import pandas as pd
df = pd.DataFrame({"first":range(200), "second":range(50,250)})

sparkDF = spark.createDataFrame(df)

newPDF = sparkDF.toPandas()
newPDF.head()
