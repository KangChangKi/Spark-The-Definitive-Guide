from pyspark.sql import SparkSession

# 참고1: https://graphframes.github.io/graphframes/docs/_site/quick-start.html
# 참고2: https://spark-packages.org/package/graphframes/graphframes
# 
# 참고2 에서 사용할 package 로 `0.8.2-spark3.2-s_2.12` 를 선택했다.

# ./bin/pyspark --packages graphframes:graphframes:0.8.2-spark3.2-s_2.12

spark = SparkSession.builder.appName("test1").master("local[*]")\
  .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")\
  .getOrCreate()

path_prefix = "file:///Users/lenani/SynologyDrive/test_personal/Spark-The-Definitive-Guide"

bikeStations = spark.read.option("header","true")\
  .csv(path_prefix + "/data/bike-data/201508_station_data.csv")
tripData = spark.read.option("header","true")\
  .csv(path_prefix + "/data/bike-data/201508_trip_data.csv")

stationVertices = bikeStations.withColumnRenamed("name", "id").distinct()
tripEdges = tripData\
  .withColumnRenamed("Start Station", "src")\
  .withColumnRenamed("End Station", "dst")

from graphframes import GraphFrame

stationGraph = GraphFrame(stationVertices, tripEdges)
stationGraph.cache()

print("Total Number of Stations: " + str(stationGraph.vertices.count()))
print("Total Number of Trips in Graph: " + str(stationGraph.edges.count()))
print("Total Number of Trips in Original Data: " + str(tripData.count()))

# Total Number of Stations: 70
# Total Number of Trips in Graph: 354152
# Total Number of Trips in Original Data: 354152

# 연산에 너무 많은 시간이 걸려서 다음과 같이 data 를 줄였다!!!
# 너무 느려서 pending 되어 있는 것 처럼 느껴질 정도다!!!
# ==> 근본원인은 data 에서 날짜 format 이 재각각이어서 발생했었던 것이었다.
# 예) `8/31/2015 23:09` 와 `11/5/2015 23:09` 처럼 자리수가 변하고 있다.

stationGraph

# GraphFrame(v:[id: string, station_id: string ... 5 more fields], e:[src: string, dst: string ... 9 more fields])

stationGraph.edges

# DataFrame[Trip ID: string, Duration: string, Start Date: string, src: string, Start Terminal: string, End Date: string, dst: string, End Terminal: string, Bike #: string, Subscriber Type: string, Zip Code: string]

stationGraph.vertices

# DataFrame[station_id: string, id: string, lat: string, long: string, dockcount: string, landmark: string, installation: string]

#########################################################################################################

from pyspark.sql.functions import desc

stationGraph.edges.groupBy("src", "dst").count().orderBy(desc("count")).show(n=10, truncate=False)

# +---------------------------------------------+----------------------------------------+-----+
# |src                                          |dst                                     |count|
# +---------------------------------------------+----------------------------------------+-----+
# |San Francisco Caltrain 2 (330 Townsend)      |Townsend at 7th                         |3748 |
# |Harry Bridges Plaza (Ferry Building)         |Embarcadero at Sansome                  |3145 |
# |2nd at Townsend                              |Harry Bridges Plaza (Ferry Building)    |2973 |
# |Townsend at 7th                              |San Francisco Caltrain 2 (330 Townsend) |2734 |
# |Harry Bridges Plaza (Ferry Building)         |2nd at Townsend                         |2640 |
# |Embarcadero at Folsom                        |San Francisco Caltrain (Townsend at 4th)|2439 |
# |Steuart at Market                            |2nd at Townsend                         |2356 |
# |Embarcadero at Sansome                       |Steuart at Market                       |2330 |
# |Townsend at 7th                              |San Francisco Caltrain (Townsend at 4th)|2192 |
# |Temporary Transbay Terminal (Howard at Beale)|San Francisco Caltrain (Townsend at 4th)|2184 |
# +---------------------------------------------+----------------------------------------+-----+
# only showing top 10 rows

# 참고로 위의 동작을 Spark SQL 로도 할 수 있다:

stationGraph.edges.createOrReplaceTempView("edges")
stationGraph.vertices.createOrReplaceTempView("vertices")

spark.sql("""
select src, dst, count(*) count
from edges
group by src, dst
order by count(*) desc
""").show(n=10, truncate=False)


stationGraph.edges\
  .where("src = 'Townsend at 7th' OR dst = 'Townsend at 7th'")\
  .groupBy("src", "dst").count()\
  .orderBy(desc("count"))\
  .show(10)

# Spark SQL version:

spark.sql("""
select src, dst, count(*) count
from edges
where src = 'Townsend at 7th' OR dst = 'Townsend at 7th'
group by src, dst
order by count(*) desc
""").show(n=10, truncate=False)

# +---------------------------------------------+---------------------------------------------+-----+
# |src                                          |dst                                          |count|
# +---------------------------------------------+---------------------------------------------+-----+
# |San Francisco Caltrain 2 (330 Townsend)      |Townsend at 7th                              |3748 |
# |Townsend at 7th                              |San Francisco Caltrain 2 (330 Townsend)      |2734 |
# |Townsend at 7th                              |San Francisco Caltrain (Townsend at 4th)     |2192 |
# |Townsend at 7th                              |Civic Center BART (7th at Market)            |1844 |
# |Civic Center BART (7th at Market)            |Townsend at 7th                              |1765 |
# |San Francisco Caltrain (Townsend at 4th)     |Townsend at 7th                              |1198 |
# |Temporary Transbay Terminal (Howard at Beale)|Townsend at 7th                              |834  |
# |Townsend at 7th                              |Harry Bridges Plaza (Ferry Building)         |827  |
# |Steuart at Market                            |Townsend at 7th                              |746  |
# |Townsend at 7th                              |Temporary Transbay Terminal (Howard at Beale)|740  |
# +---------------------------------------------+---------------------------------------------+-----+
# only showing top 10 rows


townAnd7thEdges = stationGraph.edges\
  .where("src = 'Townsend at 7th' OR dst = 'Townsend at 7th'")
subgraph = GraphFrame(stationGraph.vertices, townAnd7thEdges)


# motifs 찾기.

# GraphFrame 에서는 Neo4J 의 Cypher 언어와 유사한 도메인에 특화된 언어로 쿼리를 지정한다.
# () 는 vertex 를 의미한다.
# [] 는 edge 를 의미한다.
# ; 는 AND 연산을 의미한다.
# 예) `(a)-[ab]->(b)` 는 임의의 vertex "a" 에서 edge "ab" 를 거쳐서 edge "b" 로 간다는 것을 표현한다.
# 예) `(a)-[ab]->(b); (b)-[bc]->(c); (c)-[ca]->(a)` 는 3 개를 경로가 cycle 을 이루는 패턴을 표현한다.

motifs = stationGraph.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[ca]->(a)")
motifs.createOrReplaceTempView("motifs")

spark.sql("""
select a.id, b.id, c.id,
  ab.`Start Date`, bc.`Start Date`, ca.`Start Date`,
  ab.`End Date`, bc.`End Date`, ca.`End Date`
from motifs
where ca.`Bike #` = bc.`Bike #`
and ab.`Bike #` = bc.`Bike #`
and a.id != b.id
and b.id != c.id
""").show(truncate=False)

# 아래의 코드에서 오류를 수정했다:
# 날짜가 형식이 달라서 'MM/dd/yyyy HH:mm' => 'M/d/yyyy H:m' 로 변경해줬다.
# 이것이 연산에 걸리는 속도가 느렸었던 근본원인 이었다!!!

from pyspark.sql.functions import expr

motifs.selectExpr("*",
    "to_timestamp(ab.`Start Date`, 'M/d/yyyy H:m') as abStart",
    "to_timestamp(bc.`Start Date`, 'M/d/yyyy H:m') as bcStart",
    "to_timestamp(ca.`Start Date`, 'M/d/yyyy H:m') as caStart")\
  .where("ca.`Bike #` = bc.`Bike #`").where("ab.`Bike #` = bc.`Bike #`")\
  .where("a.id != b.id").where("b.id != c.id")\
  .where("abStart < bcStart").where("bcStart < caStart")\
  .orderBy(expr("cast(caStart as long) - cast(abStart as long)"))\
  .selectExpr("a.id", "b.id", "c.id", "ab.`Start Date`", "ca.`End Date`")\
  .limit(1).show(1, False)

# 위의 graph query 에서 지정한 vertices 와 edges 를 이용할 수 있다.
# a, b, c 는 지정했던 vertices 이다.
# ab, bc, ca 는 지정했던 edges 이다.
# 
# 위의 graph query 에서 지정하지 않았던 ac 는 존재하지 않아서 이용할 수 없다.

# Spark SQL version:

spark.sql("""
select a.id, b.id, c.id,
  ab.`Start Date`, ca.`End Date`
from (
  select *,
    to_timestamp(ab.`Start Date`, 'M/d/yyyy H:m') as abStart,
    to_timestamp(bc.`Start Date`, 'M/d/yyyy H:m') as bcStart,
    to_timestamp(ca.`Start Date`, 'M/d/yyyy H:m') as caStart
  from motifs
  where ca.`Bike #` = bc.`Bike #`
    and ab.`Bike #` = bc.`Bike #`
    and a.id != b.id
    and b.id != c.id
)
where abStart < bcStart
  and bcStart < caStart
order by cast(caStart as long) - cast(abStart as long)
limit 1
""").show(truncate=False)

# 위의 query 는 가장 빠른 경로를 구하는 것이다.
# 결과로 나온 가장 빠른 경로는 약 20분이 걸린다.

# +---------------------------------------+---------------+----------------------------------------+---------------+---------------+
# |id                                     |id             |id                                      |Start Date     |End Date       |
# +---------------------------------------+---------------+----------------------------------------+---------------+---------------+
# |San Francisco Caltrain 2 (330 Townsend)|Townsend at 7th|San Francisco Caltrain (Townsend at 4th)|5/19/2015 16:09|5/19/2015 16:33|
# +---------------------------------------+---------------+----------------------------------------+---------------+---------------+

#########################################################################################################

# PageRank 는 Larry Page 가 고안한 랭킹 알고리즘이다.
# 그냥 직관적으로 인기투표 한다고 생각하면 된다.
# 
# resetProbability=0.15 는 한표의 점수라고 생각하면 되고,
# maxIter=10 는 10번 되풀이 해서 투표한다고 생각하면 된다.

from pyspark.sql.functions import desc

ranks = stationGraph.pageRank(resetProbability=0.15, maxIter=10)
ranks.vertices.orderBy(desc("pagerank")).select("id", "pagerank").show(n=10, truncate=False)

# +----------------------------------------+------------------+
# |id                                      |pagerank          |
# +----------------------------------------+------------------+
# |San Jose Diridon Caltrain Station       |4.0515048359899986|
# |San Francisco Caltrain (Townsend at 4th)|3.351183296428578 |
# |Mountain View Caltrain Station          |2.5143907710155755|
# |Redwood City Caltrain Station           |2.3263087713711874|
# |San Francisco Caltrain 2 (330 Townsend) |2.2311442913698465|
# |Harry Bridges Plaza (Ferry Building)    |1.82511201188826  |
# |2nd at Townsend                         |1.5821217785039   |
# |Santa Clara at Almaden                  |1.5730074084907693|
# |Townsend at 7th                         |1.56845658053407  |
# |Embarcadero at Sansome                  |1.5414242087748882|
# +----------------------------------------+------------------+
# only showing top 10 rows

#########################################################################################################

# inDegrees 와 outDegrees 를 사용할 수 있다.
# 
# SNS 에서 in-degree 와 out-degree 그리고 그 비율을 사용해서 인플루언서를 찾는다고 한다.

inDeg = stationGraph.inDegrees
inDeg.orderBy(desc("inDegree")).show(n=5, truncate=False)

outDeg = stationGraph.outDegrees
outDeg.orderBy(desc("outDegree")).show(n=5, truncate=False)

# +----------------------------------------+--------+
# |id                                      |inDegree|
# +----------------------------------------+--------+
# |San Francisco Caltrain (Townsend at 4th)|34810   |
# |San Francisco Caltrain 2 (330 Townsend) |22523   |
# |Harry Bridges Plaza (Ferry Building)    |17810   |
# |2nd at Townsend                         |15463   |
# |Townsend at 7th                         |15422   |
# +----------------------------------------+--------+
# only showing top 5 rows
# 
# +---------------------------------------------+---------+
# |id                                           |outDegree|
# +---------------------------------------------+---------+
# |San Francisco Caltrain (Townsend at 4th)     |26304    |
# |San Francisco Caltrain 2 (330 Townsend)      |21758    |
# |Harry Bridges Plaza (Ferry Building)         |17255    |
# |Temporary Transbay Terminal (Howard at Beale)|14436    |
# |Embarcadero at Sansome                       |14158    |
# +---------------------------------------------+---------+
# only showing top 5 rows

degreeRatio = inDeg.join(outDeg, "id")\
  .selectExpr("id", "double(inDegree)/double(outDegree) as degreeRatio")
degreeRatio.orderBy(desc("degreeRatio")).show(n=10, truncate=False)
degreeRatio.orderBy("degreeRatio").show(n=10, truncate=False)

# +----------------------------------------+------------------+
# |id                                      |degreeRatio       |
# +----------------------------------------+------------------+
# |Redwood City Medical Center             |1.5333333333333334|
# |San Mateo County Center                 |1.4724409448818898|
# |SJSU 4th at San Carlos                  |1.3621052631578947|
# |San Francisco Caltrain (Townsend at 4th)|1.3233728710462287|
# |Washington at Kearny                    |1.3086466165413533|
# |Paseo de San Antonio                    |1.2535046728971964|
# |California Ave Caltrain Station         |1.24              |
# |Franklin at Maple                       |1.2345679012345678|
# |Embarcadero at Vallejo                  |1.2201707365495336|
# |Market at Sansome                       |1.2173913043478262|
# +----------------------------------------+------------------+
# only showing top 10 rows
# 
# +-------------------------------+------------------+
# |id                             |degreeRatio       |
# +-------------------------------+------------------+
# |Grant Avenue at Columbus Avenue|0.5180520570948782|
# |2nd at Folsom                  |0.5909488686085761|
# |Powell at Post (Union Square)  |0.6434241245136186|
# |Mezes Park                     |0.6839622641509434|
# |Evelyn Park and Ride           |0.7413087934560327|
# |Beale at Market                |0.75726761574351  |
# |Golden Gate at Polk            |0.7822270981897971|
# |Ryland Park                    |0.7857142857142857|
# |San Francisco City Hall        |0.7928849902534113|
# |Palo Alto Caltrain Station     |0.8064516129032258|
# +-------------------------------+------------------+
# only showing top 10 rows

#########################################################################################################

# BFS 를 사용할 수 있다.

stationGraph.bfs(fromExpr="id = 'Townsend at 7th'",
  toExpr="id = 'Spear at Folsom'", maxPathLength=2).show(n=10)

# +--------------------+--------------------+--------------------+
# |                from|                  e0|                  to|
# +--------------------+--------------------+--------------------+
# |{65, Townsend at ...|{913371, 663, 8/3...|{49, Spear at Fol...|
# |{65, Townsend at ...|{913265, 658, 8/3...|{49, Spear at Fol...|
# |{65, Townsend at ...|{911919, 722, 8/3...|{49, Spear at Fol...|
# |{65, Townsend at ...|{910777, 704, 8/2...|{49, Spear at Fol...|
# |{65, Townsend at ...|{908994, 1115, 8/...|{49, Spear at Fol...|
# |{65, Townsend at ...|{906912, 892, 8/2...|{49, Spear at Fol...|
# |{65, Townsend at ...|{905201, 980, 8/2...|{49, Spear at Fol...|
# |{65, Townsend at ...|{904010, 969, 8/2...|{49, Spear at Fol...|
# |{65, Townsend at ...|{903375, 850, 8/2...|{49, Spear at Fol...|
# |{65, Townsend at ...|{899944, 910, 8/2...|{49, Spear at Fol...|
# +--------------------+--------------------+--------------------+
# only showing top 10 rows

#########################################################################################################

# 연결 요소 찾기 알고리즘
# 
# 연결 요소(connected component)는 자체적인 연결을 가지고 있지만 큰 그래프에는 연결되지 않은 방향성이 없는 서브그래프 이다.
# 연결 요소 알고리즘은 방향성이 없는 그래프를 가정한다.
# ==> 비유하자면 연결 요소 = 육지(mainland) 로부터 분리된 사람이 살고 있는 섬(island) 이다.
# 
# 이 알고리즘을 연산량이 크기 때문에 실행하려면 반복 수행마다 작업 상태를 저장하는 체크포인트 디렉토리를 설정해야 한다.
# 그래서 충돌이 발생하면 작업을 마쳤던 곳에서 다시 작업을 진행할 수 있게 한다.
# 
# 참고: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.setCheckpointDir.html
# ==> spark.sparkContext.setCheckpointDir 는 다른 연산량이 많은 경우 에서도 유용하게 사용될 수 있는 좋은 tip 이다.

spark.sparkContext.setCheckpointDir(path_prefix + "/tmp/checkpoints")

minGraph = GraphFrame(stationVertices, tripEdges.sample(False, 0.1))
cc = minGraph.connectedComponents()

cc.where("component != 0").show()

# +----------+--------------------+---------+-----------+---------+-------------+------------+------------+
# |station_id|                  id|      lat|       long|dockcount|     landmark|installation|   component|
# +----------+--------------------+---------+-----------+---------+-------------+------------+------------+
# |        84|         Ryland Park|37.342725|-121.895617|       15|     San Jose|    4/9/2014|128849018880|
# |        12|SJSU 4th at San C...|37.332808|-121.883891|       19|     San Jose|    8/7/2013|128849018880|
# |        16|SJSU - San Salvad...|37.333955|-121.877349|       15|     San Jose|    8/7/2013|128849018880|
# |        80|Santa Clara Count...|37.352601|-121.905733|       15|     San Jose|  12/31/2013|128849018880|
# |        47|     Post at Kearney|37.788975|-122.403452|       19|San Francisco|   8/19/2013|317827579904|
# |        46|Washington at Kea...|37.795425|-122.404767|       15|San Francisco|   8/19/2013| 17179869184|
# |         9|           Japantown|37.348742|-121.894715|       15|     San Jose|    8/5/2013|128849018880|
# |        10|  San Jose City Hall|37.337391|-121.886995|       15|     San Jose|    8/6/2013|128849018880|
# |        14|Arena Green / SAP...|37.332692|-121.900084|       19|     San Jose|    8/5/2013|128849018880|
# |        13|       St James Park|37.339301|-121.889937|       15|     San Jose|    8/6/2013|128849018880|
# |         6|    San Pedro Square|37.336721|-121.894074|       15|     San Jose|    8/7/2013|128849018880|
# |         8| San Salvador at 1st|37.330165|-121.885831|       15|     San Jose|    8/5/2013|128849018880|
# |        11|         MLK Library|37.335885| -121.88566|       19|     San Jose|    8/6/2013|128849018880|
# |         3|San Jose Civic Ce...|37.330698|-121.888979|       15|     San Jose|    8/5/2013|128849018880|
# |         5|    Adobe on Almaden|37.331415|  -121.8932|       19|     San Jose|    8/5/2013|128849018880|
# |         4|Santa Clara at Al...|37.333988|-121.894902|       11|     San Jose|    8/6/2013|128849018880|
# |         2|San Jose Diridon ...|37.329732|-121.901782|       27|     San Jose|    8/6/2013|128849018880|
# |         7|Paseo de San Antonio|37.333798|-121.886943|       15|     San Jose|    8/7/2013|128849018880|
# +----------+--------------------+---------+-----------+---------+-------------+------------+------------+

# GraphFrame 에서 제공하는 방향성이 있는 그래프와 관련된 또 다른 알고리즘은 강한 연결 요소(strongly connected component) 이다.
# 강한 연결 요소란 방향성이 고려된 상태로 강하게 연결된 구성 요소, 즉 내부의 모든 정점 쌍 사이에 경로가 존재하는 서브그래프 이다.

scc = minGraph.stronglyConnectedComponents(maxIter=3)
scc.groupBy("component").count().show()

# +------------+-----+
# |   component|count|
# +------------+-----+
# |           0|   52|
# | 17179869184|    1|
# |128849018880|   16|
# |317827579904|    1|
# +------------+-----+
