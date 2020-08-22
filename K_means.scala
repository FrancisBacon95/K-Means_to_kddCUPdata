/////////////////////////////
////////// K-Means //////////
/////////////////////////////
/*
1) 지도학습
분류기는 찾고자 하는 것이 무엇인지 이미 알고 있고,
입력과 그에 대한 결과가 알려진 사례가 확보되었을 때만 도움이 된다.
학습 과정에서 사례별 정확한 결과값을 입력으로 받아들이기 때문에 
이런 종류의 방법을 지도학습이라 한다.

2) 비지도학습
비지도 학습에서는 이용할 수 있는 목표값이 하나도 없기 때문에
목표값을 예측하기 위한 학습을 하지 않는다.
- 그러나 이 방법은 데이터의 구조를 학습할 수 있고,
- 비슷한 입력들의 집단을 찾을 수 있고,
- 발생 가능성이 높은 입력과 그렇지 않은 입력의 종류를 학습할 수 있다.

2-1) 이상탐지
입력 데이터의 정상적인 형태를 학습해가며 
새로운 데이터가 과거 데이터와 비슷하지 않을 때 이를 감지한다.
잠재적 위험사항을 탐지. 
(이상탐지 in 비지도학습)

2-2) K-Means 군집화
군집화는 널리 알려진 비지도학습 중 하나다.
군집화 알고리즘은 데이터를 자연스럽게 군집으로 묶는 것을 목적으로 함.
- K개의 군집을 연구자가 정하고 이에 맞게 군집화한다.
-> K : 해당모델의 하이퍼파라미터
-> K를 선정하는 것이 핵심
-> 주로 유클리드 거리를 사용함(모두 숫자형 데이터인 곳에서 사용)
-> 유클리드 공간에서 각각의 요소(벡터)는 점으로 취급

군집 중심(Centroid)
:군집의 중앙. 점들의 산술평균에 해당.
-> 일부 데이터를 초기 Centroid로 선택한다.
-> 이후 모든 각각의 데이터 포인터는 가장 가까운 중심으로 할당된다.
-> 군집별로 군집에 할당된 모든 데이터의 산술 평균을 구한다.
-> 이를 통해 새로운 Centroid를 선정한다.
-> 이 과정을 반복.
*/

//////////////////////////
//// 5. 첫 번째 군집화 ////
//////////////////////////
//// 01 ////
val data = spark.read.
      option("inferSchema", true).
      option("header", false).
      csv("./kddcup.data.corrected").
      toDF(
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label")

//// 02 ////
data.select("label").groupBy("label").count().orderBy($"count".desc).show(25)
/*
+----------------+-------+
|           label|  count|
+----------------+-------+
|          smurf.|2807886|
|        neptune.|1072017|
|         normal.| 972781|
|          satan.|  15892|
|        ipsweep.|  12481|
|      portsweep.|  10413|
|           nmap.|   2316|
|           back.|   2203|
|    warezclient.|   1020|
|       teardrop.|    979|
|            pod.|    264|
|   guess_passwd.|     53|
|buffer_overflow.|     30|
|           land.|     21|
|    warezmaster.|     20|
|           imap.|     12|
|        rootkit.|     10|
|     loadmodule.|      9|
|      ftp_write.|      8|
|       multihop.|      7|
|            phf.|      4|
|           perl.|      3|
|            spy.|      2|
+----------------+-------+

sumurf., neptune. 공격빈도가 가장 높다.

- 본 데이터에는 수치형이 아닌 변수도 존재.
*/

//// 03 ////
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler

val numericOnly = data.drop("protocol_type","service","flag").cache()

val assembler = new VectorAssembler().
    setInputCols(numericOnly.columns.filter(_ != "label")).
    setOutputCol("featureVector")

val kmeans = new KMeans().
    setPredictionCol("cluster").
    setFeaturesCol("featureVector")
    
val pipeline=new Pipeline().setStages(Array(assembler,kmeans))
val pipelineModel=pipeline.fit(numericOnly)
val kmeansModel=pipelineModel.stages.last.asInstanceOf[KMeansModel]

kmeansModel.clusterCenters.foreach(println)
/*
[47.979395571029514,1622.078830816566,868.5341828266062,4.453261001578883E-5,0.006432937937735314,1.4169466823205539E-5,0.03451682118132869,1.5181571596291647E-4,0.14824703453301485,0.01021213716043885,1.1133152503947209E-4,3.6435771831099954E-5,0.011351767134933808,0.0010829521072021374,1.0930731549329986E-4,0.0010080563539937655,0.0,0.0,0.0013865835391279706,332.2862475203433,292.9071434354884,0.17668541759442963,0.17660780940042922,0.05743309987449894,0.05771839196793656,0.7915488441763401,0.020981640419415717,0.028996862475203753,232.4707319541719,188.6660459090725,0.7537812031901896,0.03090561110887087,0.6019355289259497,0.0066835148374549125,0.17675395732965926,0.17644162179668316,0.05811762681672753,0.05741111695882672]
[2.0,6.9337564E8,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,57.0,3.0,0.79,0.67,0.21,0.33,0.05,0.39,0.0,255.0,3.0,0.01,0.09,0.22,0.0,0.18,0.67,0.05,0.33]

각각의 결과들은 모델로부터 얻은 군집들의 Centroid이다.
2개 벡터가 출력됨으로 2개의 군집이 형성됐음을 알 수 있다.
데이터 내의 개별 군집을 정확하게 찾아냈다고 하기엔 불충분하다.
*/

//// 04 ////
//각각의 군집이 포함하는 레이블을 집계
val withCluster=pipelineModel.transform(numericOnly)

withCluster.select("cluster","label").
    groupBy("cluster","label").count().
    orderBy($"cluster",$"count".desc).
    show(25)
/*
+-------+----------------+------+
|cluster|           label| count|
+-------+----------------+------+
|      0|          smurf.|280790|
|      0|        neptune.|107201|
|      0|         normal.| 97278|
|      0|           back.|  2203|
|      0|          satan.|  1589|
|      0|        ipsweep.|  1247|
|      0|      portsweep.|  1039|
|      0|    warezclient.|  1020|
|      0|       teardrop.|   979|
|      0|            pod.|   264|
|      0|           nmap.|   231|
|      0|   guess_passwd.|    53|
|      0|buffer_overflow.|    30|
|      0|           land.|    21|
|      0|    warezmaster.|    20|
|      0|           imap.|    12|
|      0|        rootkit.|    10|
|      0|     loadmodule.|     9|
|      0|      ftp_write.|     8|
|      0|       multihop.|     7|
|      0|            phf.|     4|
|      0|           perl.|     3|
|      0|            spy.|     2|
|      1|      portsweep.|     1|
+-------+----------------+------+

단 하나의 데이터만 군집1에 할당되어 있으므로
역시나 의미없는 군집화로 판단할 수 있다.
*/

//////////////////////
//// 6. K 선정하기 ////
//////////////////////
//// 01 ////
import scala.util.Random
import org.apache.spark.sql.DataFrame

def clusteringScore0(data:DataFrame,k:Int): Double={
    val assembler=new VectorAssembler().
        setInputCols(data.columns.filter(_!="label")).
        setOutputCol("featureVector")
    val kmeans = new KMeans().
        setSeed(Random.nextLong()).
        setK(k).
        setPredictionCol("cluster").
        setFeaturesCol("featureVector")
    
    val pipeline=new Pipeline().setStages(Array(assembler,kmeans))
    val kmeansModel=pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
}

(20 to 100 by 20).map(k=>(k, clusteringScore0(numericOnly, k))).foreach(println)
/*
cluster 개수를 20~100까지 20간격으로 증가시킴.
이에 따른 K-means 평가점수를 확인한다.
(20,4.908678397870071E7)
(40,3.235716113221139E7)
(60,1.3508990620176084E7)
(80,1.0868998480191652E7)
(100,3.403395209149755E7)
평가 점수가 점점 낮아짐을 확인할 수 있다.

군집이 추가될수록 데이터는 항상 가장 가까운 군집 중심에 더 가까워져야 한다.
이상한 것은 군집 수가 100일 때 평가점수(데이터들의 각각의 Centroid에 대한 평균 거리)
랜덤하게 선택된 초기 Centroid로부터 반복해서 군집을 계산하다 보면
Local Minimum으로 수렴하기도 하기 때문에
최종결과가 좋을 수는 있어도 최적임을 보장할 수는 없다.
(스파크에서는 K-means II가 기본이다.)

k=100일 때 랜덤하게 선택된 초기 Centroid들의 분포로 인해,
Sub-Optimal(준최적)으로 군집화 됐을 수도 있다.
즉 전역으로 최적화되지 못하고 지역적으로 최적화된 군집을 형성했을 수도 있다.
Local Optimum(지역 최적점)에 도달하기 전에 알고리즘의 동작이 멈췄을 수도 있다.

*/

//// 02 ////
/*
이를 반복 횟수를 늘리거나 문턱값을 통해 극복할 수도 있다.
1) SetTol() 
문턱값을 하나 지정하고, Centroid가 이동하는 정도가 이 문턱값 밑으로
떨어질 때까지 반복하게 할 수 있다.
따라서 이 값이 낮을수록 군집 중심들이 이동을 멈추기까지 더 오래 걸릴 수 있다.

2) setMaxIter()
최대 반복 횟수를 늘려 너무 일찍 계산이 멈추는 것을 막을 수 있다.
*/
def clusteringScore1(data:DataFrame,k:Int): Double={

    val assembler=new VectorAssembler().
        setInputCols(data.columns.filter(_!="label")).
        setOutputCol("featureVector")
    val kmeans = new KMeans().
        setSeed(Random.nextLong()).
        setK(k).
        setPredictionCol("cluster").
        setFeaturesCol("featureVector").
        setMaxIter(40).//clusteringScore0에서 추가
        setTol(1.0e-5)//clusteringScore0에서 추가
    
    val pipeline=new Pipeline().setStages(Array(assembler,kmeans))
    val kmeansModel=pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
}

(20 to 100 by 20).map(k=>(k, clusteringScore0(numericOnly, k))).foreach(println)
/*
(20,3.253848653353835E7)
(40,3.889980509679854E7)
(60,2.733809053239971E7)
(80,4530129.88307022)
(100,3.016257788276282E7)
*/

//////////////////////////////////
//// 7. sparkR을 이용한 시각화 ////
//////////////////////////////////

//.R 파일로 작성

///////////////////
//// 8. 정규화 ////
///////////////////
import org.apache.spark.ml.feature.StandardScaler
//StandardScaler를 통해 정규화를 실시한다.

def clusteringScore2(data:DataFrame,k:Int): Double={

    val assembler=new VectorAssembler().
        setInputCols(data.columns.filter(_!="label")).
        setOutputCol("featureVector")

    //clusteringScore1에서 정규화 추가
    val scaler=new StandardScaler().
        setInputCol("featureVector").
        setOutputCol("scaledFeatureVector").
        setWithStd(true).
        setWithMean(false)

    val kmeans = new KMeans().
        setSeed(Random.nextLong()).
        setK(k).
        setPredictionCol("cluster").
        setFeaturesCol("featureVector").
        setMaxIter(40).//clusteringScore0에서 추가
        setTol(1.0e-5)//clusteringScore0에서 추가
    
    //파이프라인에 정규화 추가
    val pipeline=new Pipeline().setStages(Array(assembler,scaler,kmeans))
    val kmeansModel=pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
}

(60 to 270 by 30).map(k=>(k, clusteringScore2(numericOnly, k))).foreach(println)
/*
그 결과 차원을 더 평등하게 만드는데 도움이 됨.
점들 상의 절대거리(비용)이 훨씬 짧아졌다.
하지만 아직 K 값이 증가해도 비용이 크게 늘지 않는 명확한 지점은 확인되지 않는다.

     | (60,2.3487900978393693E7)
     | (90,3.973722412773712E7)
     | (120,3905543.6861785916)
     | (150,3762002.792157456)
     | (180,3170231.4119016505)
     | (210,717412.296098293)
     | (240,9767080.413396066)
     | (270,1807515.5062444252)
*/


///////////////////////
//// 9. 범주형 변수 ////
///////////////////////
//One-Hot Encoding
//K-means의 경우에는 수치형에만 사용할 수 있다.
//그러나 범주형 같은 경우에는 One-Hot Encoding을 통해서
//K-means에 사용가능하다.
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer().
      setInputCol(inputCol).
      setOutputCol(inputCol + "_indexed")

    val encoder = new OneHotEncoder().
      setInputCol(inputCol + "_indexed").
      setOutputCol(inputCol + "_vec")

    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, inputCol + "_vec")
  }

def clusteringScore3(data: DataFrame, k:Int):Double={
    //One-Hot Encoding 실시
    val (protoTypeEncoder, protoTypeVecCol)=oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol)=oneHotPipeline("service")
    val (flagEncoder, flagVecCol)=oneHotPipeline("flag")
    
    //데이터에서 --Seq()를 삭제 후 ++Seq()를 붙여준다.
    val assembleCols=Set(data.columns:_*) --
        Seq("label","protocol_type","service","flag")++
        Seq(protoTypeVecCol,serviceVecCol,flagVecCol)

    val assembler= new VectorAssembler().
        setInputCols(assembleCols.toArray).//이를 추가해줌
        setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline =new Pipeline().setStages(
        Array(protoTypeEncoder, serviceEncoder,flagEncoder,assembler,scaler,kmeans))

    val piplineModel=pipeline.fit(data)

    val kmeansModel=pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data))
}

//여기서는 범주형 자료들도 포함되기 때문에
//기존에 사용한 numericOnly 데이터가 아닌 최초 데이터를 이용한다.
(60 to 270 by 30).map(k=>(k, clusteringScore3(data, k))).foreach(println)

///////////////////
  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer().
      setInputCol(inputCol).
      setOutputCol(inputCol + "_indexed")
    val encoder = new OneHotEncoder().
      setInputCol(inputCol + "_indexed").
      setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, inputCol + "_vec")
  }

  def clusteringScore3(data: DataFrame, k: Int): Double = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.summary.trainingCost
  }

(60 to 270 by 30).map(k => (k, clusteringScore3(data, k))).foreach(println)
/*
(60,1.804946652342566E7)
(90,6371108.97822906)
(120,1651639.6519294935)
(150,963971.2704818384)
(180,766310.4193176146)
(210,597455.8581381304)
(240,486685.5819448269)
(270,398572.93358803826)
*/


///////////////////////////////////////
//// 10. 엔트로피와 함께 레이블 활용 ////
///////////////////////////////////////
def entropy(counts: Iterable[Int]): Double = {
    val values = counts.filter(_ > 0)
    val n = values.map(_.toDouble).sum
    values.map { v =>
      val p = v / n
      -p * math.log(p)
    }.sum
}

val clusterLabel = pipelineModel.transform(data).
    select("cluster", "label").as[(Int, String)]

val weightedClusterEntropy = clusterLabel.
    // Extract collections of labels, per cluster
    groupByKey { case (cluster, _) => cluster }.
    mapGroups { case (_, clusterLabels) =>
        val labels = clusterLabels.map { case (_, label) => label }.toSeq    
        // Count labels in collections
        val labelCounts = labels.groupBy(identity).values.map(_.size)
        labels.size * entropy(labelCounts)    
}.collect()

////////////////////
//// 11. 군집화 ////
///////////////////
//// 01 ////
import org.apache.spark.ml.PipelineModel

def fitPipeline4(data: DataFrame, k: Int): PipelineModel = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    pipeline.fit(data)
  }


def clusteringScore4(data: DataFrame, k: Int): Double = {
    val pipelineModel = fitPipeline4(data, k)

    // Predict cluster for each datum
    val clusterLabel = pipelineModel.transform(data).
        select("cluster", "label").as[(Int, String)]
    val weightedClusterEntropy = clusterLabel.
        // Extract collections of labels, per cluster
        groupByKey { case (cluster, _) => cluster }.
        mapGroups { case (_, clusterLabels) =>
        val labels = clusterLabels.map { case (_, label) => label }.toSeq
        // Count labels in collections
        val labelCounts = labels.groupBy(identity).values.map(_.size)
        labels.size * entropy(labelCounts)
        }.collect()

    // Average entropy weighted by cluster size
    weightedClusterEntropy.sum / data.count()
}


def clusteringTake4(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, clusteringScore4(data, k))).foreach(println)

    val pipelineModel = fitPipeline4(data, 180)
    val countByClusterLabel = pipelineModel.transform(data).
      select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy("cluster", "label")
    countByClusterLabel.show()
  }

clusteringTake4(data)
/*
(60,0.052684759397535065)
(90,0.04503124533451535)
(120,0.043515853740287366)
(150,0.02417533731431885)
(180,0.020011414122332455)
(210,0.027917144609656087)
(240,0.011672128236487944)
(270,0.008491531389900685)
+-------+----------+------+
|cluster|     label| count|
+-------+----------+------+
|      0|  neptune.| 82120|
|      0|portsweep.|    11|
|      1|   normal.|     6|
|      1|    smurf.|280761|
|      2|  ipsweep.|     2|
|      2|  neptune.|    99|
|      2|portsweep.|     1|
|      3|  neptune.|    92|
|      3|portsweep.|     2|
|      3|    satan.|     1|
|      4|  neptune.|   107|
|      4|portsweep.|     2|
|      5|  neptune.|   113|
|      5|portsweep.|     2|
|      6|   normal.|     1|
|      7|   normal.|   322|
|      7|    satan.|     1|
|      8|  neptune.|    88|
|      8|portsweep.|     1|
|      9|  neptune.|    89|
+-------+----------+------+
only showing top 20 rows
*/

/*
clusteringScore4를 이용해 이상탐지 시스템을 생성한다.
이상탐지는 새로운 데이터에서 가장 가까운 군집 중심과의 거리를 측정하는 방식으로 진행된다.
이 거리가 문턱값을 넘어서는 데이터는 이상한 것으로 판단한다.

여기서는 기존 데이터 중 군집중심으로부터 100번째 먼 데이터와의 거리를
문턱값으로 설정한다.
*/
//// 02 ////
import org.apache.spark.ml.linalg.{Vector, Vectors}
def buildAnomalyDetector(data: DataFrame): Unit = {
    val pipelineModel = fitPipeline4(data, 180)

    val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    val centroids = kMeansModel.clusterCenters

    val clustered = pipelineModel.transform(data)
    val threshold = clustered.
      select("cluster", "scaledFeatureVector").as[(Int, Vector)].
      map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }.
      orderBy($"value".desc).take(100).last

    val originalCols = data.columns
    val anomalies = clustered.filter { row =>
      val cluster = row.getAs[Int]("cluster")
      val vec = row.getAs[Vector]("scaledFeatureVector")
      Vectors.sqdist(centroids(cluster), vec) >= threshold
      //sqdist : 제곱거리
    }.select(originalCols.head, originalCols.tail:_*)

    println(anomalies.show())
  }
//중간 중간에 normal.이 있긴 하나 이상감지가 되고 있는 것을 확인할 수 있다.
