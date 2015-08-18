package com.recipe

import io.prediction.controller.{P2LAlgorithm, Params}
import io.prediction.data.storage.{BiMap, Event}
import io.prediction.data.store.LEventStore

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.{Rating => MLlibRating}
import org.apache.spark.mllib.feature.{Normalizer, StandardScaler}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

import scala.reflect.ClassTag
import scala.collection.mutable.PriorityQueue
import scala.concurrent.duration.Duration
import scala.concurrent.ExecutionContext.Implicits.global

/**
 * --알고리즘 초기설정 파라미터--
 * appName: predictionIO 앱 이름
 * unseenOnly: unseen 이벤트만 보여줌
 * seenEvents: 유저가 본 이벤트의 user-to-item 리스트, unseenOnly가 true일 때 쓰임
 * similarEvents: 비슷한 이벤트의 user-item-item 리스트, 유저가 최근에 본 item과 비슷한 item을 찾을 때 쓰임
 * rank: MLlib ALS 알고리즘의 파라미터. Number of latent feature.
 * numIterations: MLlib ALS 알고리즘의 파라미터. Number of iterations.
 * lambda: MLlib ALS 알고리즘의 정규화 파라미터
 * seed: MLlib ALS 알고리즘의 random seed. (Optional)
 * dimension: 벡터화 된 아이템의 차원 수
 * cooktimeWeight: 조리시간의 가중치
 * caloriesWeight: 칼로리의 가중치
 * expireWeight: 보관기간의 가중치
 * normalizeProjection: projection 표준화
 */
case class RecipeAlgorithmParams(
  appName: String,
  unseenOnly: Boolean,
  seenEvents: List[String],
  similarEvents: List[String],
  rank: Int,
  numIterations: Int,
  lambda: Double,
  seed: Option[Long],
  dimensions: Int,
  cooktimeWeight: Double,
  caloriesWeight: Double,
  expireWeight: Double,
  normalizeProjection: Boolean
) extends Params

/**
 * --레시피 모델--
 * item: 레시피
 * features: ALS 알고리즘으로 계산된 score
 * count: similar product가 없을 때 trainDefault()에 의해 반환된 popular count score
 */
case class RecipeModel(
  item: Item,
  features: Option[Array[Double]], // features by ALS
  count: Int // popular count for default score
)

/**
 * --레시피 알고리즘 모델--
 * rank: MLlib ALS 알고리즘의 파라미터. Number of latent feature.
 * userFreatures: 유저의 최근 행동 기록
 * recipeModels: 레시피 모델(item, features, count)
 * userStringIntMap: 유저String을 Int로 Mapping
 * itemStringIntMap: 아이템String을 Int로 Mapping
 * itemIds: 아이템id
 * projection: projection 매트릭스
 */
class RecipeAlgorithmModel(
  val rank: Int,
  val userFeatures: Map[Int, Array[Double]],
  val recipeModels: Map[Int, RecipeModel],
  val userStringIntMap: BiMap[String, Int],
  val itemStringIntMap: BiMap[String, Int],
  val itemIds: BiMap[String, Int], 
  val projection: DenseMatrix
) extends Serializable {

  @transient lazy val itemIntStringMap = itemStringIntMap.inverse

  override def toString = {
    s" rank: ${rank}" +
    s" userFeatures: [${userFeatures.size}]" +
    s"(${userFeatures.take(2).toList}...)" +
    s" recipeModels: [${recipeModels.size}]" +
    s"(${recipeModels.take(2).toList}...)" +
    s" userStringIntMap: [${userStringIntMap.size}]" +
    s"(${userStringIntMap.take(2).toString}...)]" +
    s" itemStringIntMap: [${itemStringIntMap.size}]" +
    s"(${itemStringIntMap.take(2).toString}...)]" +
    s"Items: ${itemIds.size}"
  }
}

/**
 * --레시피 알고리즘--
 * Collaborative Filtering과 Content Based Filtering방식을 독립적으로 시행한 뒤 score를 합산함
 */
class RecipeAlgorithm(val ap: RecipeAlgorithmParams)
  extends P2LAlgorithm[PreparedData, RecipeAlgorithmModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): RecipeAlgorithmModel = {
    /* Collaborative Filtering */
    require(!data.viewEvents.take(1).isEmpty,
      s"viewEvents in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
    require(!data.likeEvents.take(1).isEmpty,
      s"likeEvents in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
    require(!data.users.take(1).isEmpty,
      s"users in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
    require(!data.items.take(1).isEmpty,
      s"items in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")

    // User와 Item의 String ID를 integer index BiMap으로 생성
    val userStringIntMap = BiMap.stringInt(data.users.keys)
    val itemStringIntMap = BiMap.stringInt(data.items.keys)

    val mllibRatings: RDD[MLlibRating] = genMLlibRating(
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      data = data
    )

    // 만약 training data가 없으면 MLLib ALS 알고리즘을 수행할 수 없음
    require(!mllibRatings.take(1).isEmpty,
      s"mllibRatings cannot be empty." +
      " Please check if your events contain valid user and item ID.")

    // MLlib ALS 알고리즘의 seed
    val seed = ap.seed.getOrElse(System.nanoTime)

    // featrure vector들을 training 하기 위해 ALS의 암묵적선호도 알고리즘을 사용함
    val m = ALS.trainImplicit(
      ratings = mllibRatings,
      rank = ap.rank,
      iterations = ap.numIterations,
      lambda = ap.lambda,
      blocks = -1,
      alpha = 1.0,
      seed = seed)

    val userFeatures = m.userFeatures.collectAsMap.toMap

    // item String을 Int로 변환
    val items = data.items.map { case (id, item) =>
      (itemStringIntMap(id), item)
    }

    // item과 trained productFeature를 Mapping
    val productFeatures: Map[Int, (Item, Option[Array[Double]])] =
      items.leftOuterJoin(m.productFeatures).collectAsMap.toMap

    // 유저 정보가 없을시 다른 유저의 view 이벤트와 like 이벤트를 기준으로 합산한 popular count를 training에 이용
    val popularCount = trainDefault(
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      data = data
    )

    val recipeModels: Map[Int, RecipeModel] = productFeatures
      .map { case (index, (item, features)) =>
        val pm = RecipeModel(
          item = item,
          features = features,
          // NOTE: popularCount는 모든 아이템을 포함하지 않을 수 있으니 getOrElse를 사용함
          count = popularCount.getOrElse(index, 0)
        )
        (index, pm)
      }




    /* Content Based Filtering */
    val itemIds = BiMap.stringInt(data.items.map(_._1))

    // categorical var을 one-hot encoding을 이용해 encoding함
    val categorical = Seq(encode(data.items.map(_._2.categories)),
      encode(data.items.map(_._2.feelings)))

    /**
     * numeric var를 변환함.
     * categorical attribute들이 binary로 encoding되었기 때문에 numeric var를 
     * binary로 scaling한 후 미리 설정한 가중치가 주어짐.
     * 이 가중치는 feature별로 중요도에 따라 다르게 설정 가능함
     */
    val numericRow = data.items.map(x => Vectors.dense(x._2.cooktime, x._2
      .calories,x._2.expire))
    val weights = Array(ap.cooktimeWeight, ap.caloriesWeight, ap.expireWeight)
    val scaler = new StandardScaler(withMean = true,
      withStd = true).fit(numericRow)
    val numeric = numericRow.map(x => Vectors.dense(scaler.transform(x).
      toArray.zip(weights).map { case (x, w) => x * w }))

    /**
     * 모든 data를 병합한 후 표준화 함.
     * 이를 통해 벡터간의 코사인 값을 구할 수 있음
     */
    val normalizer = new Normalizer(p = 2.0)
    val allData = normalizer.transform((categorical ++ Seq(numeric)).reduce(merge))

    /**
     * Now we need to transpose RDD because SVD better works with ncol << nrow
     * and it's often the case when number of binary attributes is much greater
     * then the number of items. But in the case when the number of items is
     * more than number of attributes it is better not to transpose. In such
     * case U matrix should be used
     */

    /**
     * SVD 는 row의 갯수가 많을 때 더 효율적으로 작동하므로 RDD를 transpose 할 필요가 있음.
     * 일반적으로 
     */
    val transposed = transposeRDD(allData)

    val mat: RowMatrix = new RowMatrix(transposed)

    // Make SVD to reduce data dimensionality
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(
      ap.dimensions, computeU = false)

    val V: DenseMatrix = new DenseMatrix(svd.V.numRows, svd.V.numCols,
      svd.V.toArray)

    val projection = Matrices.diag(svd.s).multiply(V.transpose)

/*
    // This is an alternative code for the case when data matrix is not
    // transposed (when the number of items is much bigger then the number
    // of binary attributes

    val mat: RowMatrix = new RowMatrix(allData)

    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(
      ap.dimensions, computeU = true)

    val U: DenseMatrix = new DenseMatrix(svd.U.numRows.toInt, svd.U.numCols
      .toInt, svd.U.rows.flatMap(_.toArray).collect(), isTransposed = true)

    val projection = Matrices.diag(svd.s).multiply(U.transpose)
*/

    svd.s.toArray.zipWithIndex.foreach { case (x, y) =>
      logger.info(s"Singular value #$y = $x") }

    val maxRank = Seq(mat.numCols(), mat.numRows()).min
    val total = svd.s.toArray.map(x => x * x).reduce(_ + _)
    val worstLeft = svd.s.toArray.last * svd.s.toArray.last * (maxRank - svd.s.size)
    val variabilityGrasped = 100 * total / (total + worstLeft)

    logger.info(s"Worst case variability grasped: $variabilityGrasped%")

    val res = if(ap.normalizeProjection) {
      val sequentionalizedProjection = for (j <- 0 until projection.numCols)
        yield Vectors.dense((for (i <- 0 until projection.numRows) yield
        projection(i, j)).toArray)

      val normalizedProjectionSeq = sequentionalizedProjection.map(x =>
        normalizer.transform(x))

      val normalizedProjection = new DenseMatrix(projection.numRows, projection
        .numCols, normalizedProjectionSeq.flatMap(x => x.toArray).toArray)

      new RecipeAlgorithmModel(
      rank = m.rank,
      userFeatures = userFeatures,
      recipeModels = recipeModels,
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      itemIds = itemIds,
      projection = normalizedProjection
      )
    } else {
      new RecipeAlgorithmModel(
      rank = m.rank,
      userFeatures = userFeatures,
      recipeModels = recipeModels,
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      itemIds = itemIds,
      projection = projection
      )
    }

    res
  }

  /* Generate MLlibRating from PreparedData. */
  def genMLlibRating(
    userStringIntMap: BiMap[String, Int],
    itemStringIntMap: BiMap[String, Int],
    data: PreparedData): RDD[MLlibRating] = {

    val mllibRatings1 = data.viewEvents
      .map { r =>
        // Convert user and item String IDs to Int index for MLlib
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        ((uindex, iindex), 1)
      }
      .filter { case ((u, i), v) =>
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .reduceByKey(_ + _) // aggregate all view events of same user-item pair

    val mllibRatings2 = data.likeEvents
      .map { r =>
        // Convert user and item String IDs to Int index for MLlib
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        // key is (uindex, iindex) tuple, value is (like, t) tuple
        ((uindex, iindex), (r.like, r.t))
      }.filter { case ((u, i), v) =>
        // val  = d
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }.reduceByKey { case (v1, v2) => 
        // An user may like an item and change to cancel_like it later,
        // or vice versa. Use the latest value for this case.
        val (like1, t1) = v1
        val (like2, t2) = v2
        // keep the latest value
        if (t1 > t2) v1 else v2
      }.map { case ((u, i), (like, t)) => 
        // We use ALS.trainImplicit()
        val r = if (like) 5 else 0  // the rating of "like" event is 5, "cancel_like" event is 0
        ((u, i), r)
      }


      val sumOfmllibRatings = mllibRatings1.union(mllibRatings2).reduceByKey(_ + _)
        .map { case ((u, i), v) =>
        // MLlibRating requires integer index for user and item
        MLlibRating(u, i, v)
      }.cache()

      sumOfmllibRatings
  }

  /** 
   * Add all likes and views of each items for scoring.
   * This is for Default Prediction when know nothing about the user.
   */
  def trainDefault(
    userStringIntMap: BiMap[String, Int],
    itemStringIntMap: BiMap[String, Int],
    data: PreparedData): Map[Int, Int] = {
    
    // count number of likes
    // (item index, count)
    val likeCountsRDD: RDD[(Int, Int)] = data.likeEvents
      .map { r =>
        // Convert user and item String IDs to Int index
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        // key is (uindex, iindex) tuple, value is (like, t) tuple
        ((uindex, iindex), (r.like, r.t))
      }
      .filter { case ((u, i), v) =>
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .map { case ((u, i), (like, t)) => 
        if (like) (i, 5) else (i, -5) // like: 5, cancel_like: -5
      } // key is item
      .reduceByKey(_ + _) // aggregate all like events of same user-item pair


    // count number of views
    // (item index, count)
    val viewCountsRDD: RDD[(Int, Int)] = data.viewEvents
      .map { r =>
        // Convert user and item String IDs to Int index
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        ((uindex, iindex), 1) // view: 1
      }
      .filter { case ((u, i), v) =>
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .map { case ((u, i), v) => 
        (i, v);
      } // key is item
      .reduceByKey(_ + _) // aggregate all view events of same user-item pair
    
    // aggregate all like and view events of same user-item pair
    val sumOfAll: RDD[(Int, Int)] = likeCountsRDD.union(viewCountsRDD).reduceByKey(_ + _)

    sumOfAll.collectAsMap.toMap
  }

  def predict(model: RecipeAlgorithmModel, query: Query): PredictedResult = {
    /* Collaborative Filtering */
    val userFeatures = model.userFeatures
    val recipeModels = model.recipeModels

    // convert whiteList's string ID to integer index
    val whiteList: Option[Set[Int]] = query.whiteList.map( set =>
      set.flatMap(model.itemStringIntMap.get(_))
    )

    val finalBlackList: Set[Int] = genBlackList(query = query)
      // convert seen Items list from String ID to interger Index
      .flatMap(x => model.itemStringIntMap.get(x))

    val userFeature: Option[Array[Double]] =
      model.userStringIntMap.get(query.user).flatMap { userIndex =>
        userFeatures.get(userIndex)
      }

    val topScores: Array[(Int, Double)] = if (userFeature.isDefined) {
      // the user has feature vector
      predictKnownUser(
        userFeature = userFeature.get,
        recipeModels = recipeModels,
        query = query,
        whiteList = whiteList,
        blackList = finalBlackList
      )
    } else {
      // the user doesn't have feature vector.
      // For example, new user is created after model is trained.
      logger.info(s"No userFeature found for user ${query.user}.")

      // check if the user has recent events on some items
      val recentItems: Set[String] = getRecentItems(query)
      val recentList: Set[Int] = recentItems.flatMap (x =>
        model.itemStringIntMap.get(x))

      val recentFeatures: Set[Array[Double]] = recentList
        // recipeModels may not contain the requested item
        .map { i =>
          recipeModels.get(i).flatMap { pm => pm.features }
        }.flatten

      if (recentFeatures.isEmpty) {
        logger.info(s"No features set for recent items ${recentItems}.")
        predictDefault(
          recipeModels = recipeModels,
          query = query,
          whiteList = whiteList,
          blackList = finalBlackList
        )
      } else {
        predictSimilar(
          recentFeatures = recentFeatures,
          recipeModels = recipeModels,
          query = query,
          whiteList = whiteList,
          blackList = finalBlackList
        )
      }
    }

    val itemScores = topScores.map { case (i, s) =>
      new ItemScore(
        // convert item int index back to string ID
        item = model.itemIntStringMap(i),
        score = s
      )
    }





    /* Content Based Filtering */
    /**
     * Here we compute similarity to group of items in very simple manner
     * We just take top scored items for all query items
     */
    val recentItems: Set[String] = getRecentItems(query) 

    logger.info(recentItems) // test

    val result = recentItems.flatMap { itemId =>
      model.itemIds.get(itemId).map { j =>
        val d = for(i <- 0 until model.projection.numRows) yield model.projection(i, j)
        val col = model.projection.transpose.multiply(new DenseVector(d.toArray))
        for(k <- 0 until col.size) yield new ItemScore(model.itemIds.inverse
          .getOrElse(k, default="NA"), col(k))
      }.getOrElse(Seq())
    }.groupBy {
      case(ItemScore(itemId, _)) => itemId
    }.map(_._2.max).filter {
      case(ItemScore(itemId, _)) => !recentItems.contains(itemId)
    }.toArray.sorted.reverse.take(query.num)

    if(result.isEmpty) logger.info(s"The user has no recent action.")


    val combinedResult = itemScores.union(result) // Collaborative Filtering 결과값과 Content Based 결과값을 합침

    PredictedResult(combinedResult)
  }

  /** Generate final blackList based on other constraints */
  def genBlackList(query: Query): Set[String] = {
    // if unseenOnly is True, get all seen items
    val seenItems: Set[String] = if (ap.unseenOnly) {

      // get all user item events which are considered as "seen" events
      val seenEvents: Iterator[Event] = try {
        LEventStore.findByEntity(
          appName = ap.appName,
          entityType = "user",
          entityId = query.user,
          eventNames = Some(ap.seenEvents),
          targetEntityType = Some(Some("item")),
          // set time limit to avoid super long DB access
          timeout = Duration(200, "millis")
        )
      } catch {
        case e: scala.concurrent.TimeoutException =>
          logger.error(s"Timeout when read seen events." +
            s" Empty list is used. ${e}")
          Iterator[Event]()
        case e: Exception =>
          logger.error(s"Error when read seen events: ${e}")
          throw e
      }

      seenEvents.map { event =>
        try {
          event.targetEntityId.get
        } catch {
          case e => {
            logger.error(s"Can't get targetEntityId of event ${event}.")
            throw e
          }
        }
      }.toSet
    } else {
      Set[String]()
    }

    // get the latest constraint unavailableItems $set event
    val unavailableItems: Set[String] = try {
      val constr = LEventStore.findByEntity(
        appName = ap.appName,
        entityType = "constraint",
        entityId = "unavailableItems",
        eventNames = Some(Seq("$set")),
        limit = Some(1),
        latest = true,
        timeout = Duration(200, "millis")
      )
      if (constr.hasNext) {
        constr.next.properties.get[Set[String]]("items")
      } else {
        Set[String]()
      }
    } catch {
      case e: scala.concurrent.TimeoutException =>
        logger.error(s"Timeout when read set unavailableItems event." +
          s" Empty list is used. ${e}")
        Set[String]()
      case e: Exception =>
        logger.error(s"Error when read set unavailableItems event: ${e}")
        throw e
    }

    // combine query's blackList,seenItems and unavailableItems
    // into final blackList.
    query.blackList.getOrElse(Set[String]()) ++ seenItems ++ unavailableItems
  }

  /** Get recent events of the user on items for recommending similar items */
  def getRecentItems(query: Query): Set[String] = {
    // get latest 10 user view item events
    val recentEvents = try {
      LEventStore.findByEntity(
        appName = ap.appName,
        // entityType and entityId is specified for fast lookup
        entityType = "user",
        entityId = query.user,
        eventNames = Some(ap.similarEvents),
        targetEntityType = Some(Some("item")),
        limit = Some(10), // Default: 10 items
        latest = true,
        // set time limit to avoid super long DB access
        timeout = Duration(200, "millis")
      )
    } catch {
      case e: scala.concurrent.TimeoutException =>
        logger.error(s"Timeout when read recent events." +
          s" Empty list is used. ${e}")
        Iterator[Event]()
      case e: Exception =>
        logger.error(s"Error when read recent events: ${e}")
        throw e
    }

    val recentItems: Set[String] = recentEvents.map { event =>
      try {
        event.targetEntityId.get
      } catch {
        case e => {
          logger.error("Can't get targetEntityId of event ${event}.")
          throw e
        }
      }
    }.toSet

    recentItems
  }

  /** Prediction for user with known feature vector */
  def predictKnownUser(
    userFeature: Array[Double],
    recipeModels: Map[Int, RecipeModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = recipeModels.par // convert to parallel collection
      .filter { case (i, pm) =>
        pm.features.isDefined &&
        isCandidateItem(
          i = i,
          item = pm.item,
          categories = query.categories,
          whiteList = whiteList,
          blackList = blackList
        )
      }
      .map { case (i, pm) =>
        // NOTE: features must be defined, so can call .get
        val s = dotProduct(userFeature, pm.features.get)
        // may customize here to further adjust score
        (i, s)
      }
      .filter(_._2 > 0) // only keep items with score > 0
      .seq // convert back to sequential collection

    val ord = Ordering.by[(Int, Double), Double](_._2).reverse
    val topScores = getTopN(indexScores, query.num)(ord).toArray

    topScores
  }

  /** Default prediction when know nothing about the user */
  def predictDefault(
    recipeModels: Map[Int, RecipeModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = recipeModels.par // convert back to sequential collection
      .filter { case (i, pm) =>
        isCandidateItem(
          i = i,
          item = pm.item,
          categories = query.categories,
          whiteList = whiteList,
          blackList = blackList
        )
      }
      .map { case (i, pm) =>
        // may customize here to further adjust score
        (i, pm.count.toDouble)
      }
      .seq

    val ord = Ordering.by[(Int, Double), Double](_._2).reverse
    val topScores = getTopN(indexScores, query.num)(ord).toArray

    topScores
  }

  /** Return top similar items based on items user recently has action on (Default: 10 recent action)*/
  def predictSimilar(
    recentFeatures: Set[Array[Double]],
    recipeModels: Map[Int, RecipeModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = recipeModels.par // convert to parallel collection
      .filter { case (i, pm) =>
        pm.features.isDefined &&
        isCandidateItem(
          i = i,
          item = pm.item,
          categories = query.categories,
          whiteList = whiteList,
          blackList = blackList
        )
      }
      .map { case (i, pm) =>
        val s = recentFeatures.map{ rf =>
          // pm.features must be defined because of filter logic above
          cosine(rf, pm.features.get)
        }.reduce(_ + _)
        // may customize here to further adjust score
        (i, s)
      }
      .filter(_._2 > 0) // keep items with score > 0
      .seq // convert back to sequential collection

    val ord = Ordering.by[(Int, Double), Double](_._2).reverse
    val topScores = getTopN(indexScores, query.num)(ord).toArray

    topScores
  }

  private
  def getTopN[T](s: Iterable[T], n: Int)(implicit ord: Ordering[T]): Seq[T] = {

    val q = PriorityQueue()

    for (x <- s) {
      if (q.size < n)
        q.enqueue(x)
      else {
        // q is full
        if (ord.compare(x, q.head) < 0) {
          q.dequeue()
          q.enqueue(x)
        }
      }
    }

    q.dequeueAll.toSeq.reverse
  }

  private
  def dotProduct(v1: Array[Double], v2: Array[Double]): Double = {
    val size = v1.size
    var i = 0
    var d: Double = 0
    while (i < size) {
      d += v1(i) * v2(i)
      i += 1
    }
    d
  }

  private
  def cosine(v1: Array[Double], v2: Array[Double]): Double = {
    val size = v1.size
    var i = 0
    var n1: Double = 0
    var n2: Double = 0
    var d: Double = 0
    while (i < size) {
      n1 += v1(i) * v1(i)
      n2 += v2(i) * v2(i)
      d += v1(i) * v2(i)
      i += 1
    }
    val n1n2 = (math.sqrt(n1) * math.sqrt(n2))
    if (n1n2 == 0) 0 else (d / n1n2)
  }

  private
  def isCandidateItem(
    i: Int,
    item: Item,
    categories: Option[Set[String]],
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Boolean = {
    // whiteList와 blackList 필터링
    whiteList.map(_.contains(i)).getOrElse(true) &&
    !blackList.contains(i) &&
    // categories 필터링
    categories.map { cat =>
      item.categories.toList.headOption.map { itemCat =>
        // 쿼리의 categories와 겹치면 이 item을 keep
        !(itemCat.toSet.intersect(cat.toSet[Any]).isEmpty)
      }.getOrElse(false) // 만약 category가 없으면 item을 버림
    }.getOrElse(true)

  }

  private def encode(data: RDD[Array[String]]): RDD[Vector] = {
    val dict = BiMap.stringLong(data.flatMap(x => x))
    val len = dict.size

    data.map { sample =>
      val indexes = sample.map(dict(_).toInt).sorted
      Vectors.sparse(len, indexes, Array.fill[Double](indexes.length)(1.0))
    }
  }

  // [X: ClassTag] - 다른 input의 encode 정의를 위한 trick
  private def encode[X: ClassTag](data: RDD[String]): RDD[Vector] = {
    val dict = BiMap.stringLong(data)
    val len = dict.size

    data.map { sample =>
      val index = dict(sample).toInt
      Vectors.sparse(len, Array(index), Array(1.0))
    }
  }

  private def merge(v1: RDD[Vector], v2: RDD[Vector]): RDD[Vector] = {
    v1.zip(v2) map {
      case (SparseVector(leftSz, leftInd, leftVals), SparseVector(rightSz,
      rightInd, rightVals)) =>
        Vectors.sparse(leftSz + rightSz, leftInd ++ rightInd.map(_ + leftSz),
          leftVals ++ rightVals)
      case (SparseVector(leftSz, leftInd, leftVals), DenseVector(rightVals)) =>
        Vectors.sparse(leftSz + rightVals.length, leftInd ++ (0 until rightVals
          .length).map(_ + leftSz), leftVals ++ rightVals)
    }
  }

  private def transposeRDD(data: RDD[Vector]) = {
    val len = data.count().toInt

    val byColumnAndRow = data.zipWithIndex().flatMap {
      case (rowVector, rowIndex) => { rowVector match {
        case SparseVector(_, columnIndices, values) =>
          values.zip(columnIndices)
        case DenseVector(values) =>
          values.zipWithIndex
      }} map {
        case(v, columnIndex) => columnIndex -> (rowIndex, v)
      }
    }

    val byColumn = byColumnAndRow.groupByKey().sortByKey().values

    val transposed = byColumn.map {
      indexedRow =>
        val all = indexedRow.toArray.sortBy(_._1)
        val significant = all.filter(_._2 != 0)
        Vectors.sparse(len, significant.map(_._1.toInt), significant.map(_._2))
    }

    transposed
  }
}