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
  amountWeight: Double,
  expireWeight: Double,
  normalizeProjection: Boolean
) extends Params

case class ProductModel(
  item: Item,
  features: Option[Array[Double]], // features by ALS
  count: Int // popular count for default score
)

class RecipeModel(
  val rank: Int,
  val userFeatures: Map[Int, Array[Double]],
  val productModels: Map[Int, ProductModel],
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
    s" productModels: [${productModels.size}]" +
    s"(${productModels.take(2).toList}...)" +
    s" userStringIntMap: [${userStringIntMap.size}]" +
    s"(${userStringIntMap.take(2).toString}...)]" +
    s" itemStringIntMap: [${itemStringIntMap.size}]" +
    s"(${itemStringIntMap.take(2).toString}...)]" +
    s"Items: ${itemIds.size}"
  }
}

class RecipeAlgorithm(val ap: RecipeAlgorithmParams)
  extends P2LAlgorithm[PreparedData, RecipeModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): RecipeModel = {
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
    // create User and item's String ID to integer index BiMap
    val userStringIntMap = BiMap.stringInt(data.users.keys)
    val itemStringIntMap = BiMap.stringInt(data.items.keys)

    val mllibRatings: RDD[MLlibRating] = genMLlibRating(
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      data = data
    )

    // MLLib ALS cannot handle empty training data.
    require(!mllibRatings.take(1).isEmpty,
      s"mllibRatings cannot be empty." +
      " Please check if your events contain valid user and item ID.")

    // seed for MLlib ALS
    val seed = ap.seed.getOrElse(System.nanoTime)

    // use ALS to train feature vectors
    val m = ALS.trainImplicit(
      ratings = mllibRatings,
      rank = ap.rank,
      iterations = ap.numIterations,
      lambda = ap.lambda,
      blocks = -1,
      alpha = 1.0,
      seed = seed)

    val userFeatures = m.userFeatures.collectAsMap.toMap

    // convert ID to Int index
    val items = data.items.map { case (id, item) =>
      (itemStringIntMap(id), item)
    }

    // join item with the trained productFeatures
    val productFeatures: Map[Int, (Item, Option[Array[Double]])] =
      items.leftOuterJoin(m.productFeatures).collectAsMap.toMap

    val popularCount = trainDefault(
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      data = data
    )

    val productModels: Map[Int, ProductModel] = productFeatures
      .map { case (index, (item, features)) =>
        val pm = ProductModel(
          item = item,
          features = features,
          // NOTE: use getOrElse because popularCount may not contain all items.
          count = popularCount.getOrElse(index, 0)
        )
        (index, pm)
      }




    /* ContentBasedModel */
    val itemIds = BiMap.stringInt(data.items.map(_._1))

    /**
     * Encode categorical vars
     * We use here one-hot encoding
     */

    val categorical = Seq(encode(data.items.map(_._2.categories)),
      encode(data.items.map(_._2.feelings)))

    /**
     * Transform numeric vars.
     * In our case categorical attributes are binary encoded. Numeric vars are
     * scaled and additional weights are given to them. These weights should be
     * selected from some a-priory information, i.e. one should check how
     * important year or duration for model quality and then assign weights
     * accordingly
     */

    val numericRow = data.items.map(x => Vectors.dense(x._2.cooktime, x._2
      .amount,x._2.expire))
    val weights = Array(ap.cooktimeWeight, ap.amountWeight, ap.expireWeight)
    val scaler = new StandardScaler(withMean = true,
      withStd = true).fit(numericRow)
    val numeric = numericRow.map(x => Vectors.dense(scaler.transform(x).
      toArray.zip(weights).map { case (x, w) => x * w }))

    /**
     * Now we merge all data and normalize vectors so that they have unit norm
     * and their dot product would yield cosine between vectors
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

      new RecipeModel(
      rank = m.rank,
      userFeatures = userFeatures,
      productModels = productModels,
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      itemIds = itemIds,
      projection = normalizedProjection
      )
    } else {
      new RecipeModel(
      rank = m.rank,
      userFeatures = userFeatures,
      productModels = productModels,
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      itemIds = itemIds,
      projection = projection
      )
    }

    res
  }

  /** Generate MLlibRating from PreparedData.
    * You may customize this function if use different events or different aggregation method
    */
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

    // ADDED
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

        // MODIFIED, key is (uindex, iindex) tuple, value is (like, t) tuple
        ((uindex, iindex), (r.like, r.t))
      }.filter { case ((u, i), v) =>
        //val  = d
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }.reduceByKey { case (v1, v2) => // MODIFIED
        // An user may like an item and change to cancel_like it later,
        // or vice versa. Use the latest value for this case.
        val (like1, t1) = v1
        val (like2, t2) = v2
        // keep the latest value
        if (t1 > t2) v1 else v2
      }.map { case ((u, i), (like, t)) => // MODIFIED
        // We use ALS.trainImplicit()
        val r = if (like) 5 else 0  // MODIFIED, the rating of "like" event is 5, "cancel_like" event is 0
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
   * MODIFIED, add all likes and views of each items for scoring.
   * This is for Default Prediction when know nothing about the user.
   */
  def trainDefault(
    userStringIntMap: BiMap[String, Int],
    itemStringIntMap: BiMap[String, Int],
    data: PreparedData): Map[Int, Int] = {
    
    // MODIFIED, count number of likes
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

        // MODIFIED, key is (uindex, iindex) tuple, value is (like, t) tuple
        ((uindex, iindex), (r.like, r.t))
      }
      .filter { case ((u, i), v) =>
        // MODIFIED, keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .map { case ((u, i), (like, t)) => 
        if (like) (i, 5) else (i, -5) // like: 5, cancel_like: -5
      } // MODIFIED, key is item
      .reduceByKey(_ + _) // aggregate all like events of same user-item pair


    // MODIFIED, count number of views
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

        ((uindex, iindex), 1)
      }
      .filter { case ((u, i), v) =>
        // MODIFIED, keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .map { case ((u, i), v) => 
        (i, v);
      } // MODIFIED, key is item
      .reduceByKey(_ + _) // aggregate all view events of same user-item pair
    
    // aggregate all like and view events of same user-item pair
    val sumOfAll: RDD[(Int, Int)] = likeCountsRDD.union(viewCountsRDD).reduceByKey(_ + _)

    sumOfAll.collectAsMap.toMap
  }

  def predict(model: RecipeModel, query: Query): PredictedResult = {

    val userFeatures = model.userFeatures
    val productModels = model.productModels

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
        productModels = productModels,
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
        // productModels may not contain the requested item
        .map { i =>
          productModels.get(i).flatMap { pm => pm.features }
        }.flatten

      if (recentFeatures.isEmpty) {
        logger.info(s"No features set for recent items ${recentItems}.")
        predictDefault(
          productModels = productModels,
          query = query,
          whiteList = whiteList,
          blackList = finalBlackList
        )
      } else {
        predictSimilar(
          recentFeatures = recentFeatures,
          productModels = productModels,
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





    /* ContentBasedFiltering */
    /**
     * Here we compute similarity to group of items in very simple manner
     * We just take top scored items for all query items

     * It is possible to use other grouping functions instead of max
     */
    
    val recentItems: Set[String] = getRecentItems(query) // ADDED

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


    val combinedResult = itemScores.union(result)

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
        limit = Some(10),
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
    productModels: Map[Int, ProductModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = productModels.par // convert to parallel collection
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
    productModels: Map[Int, ProductModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = productModels.par // convert back to sequential collection
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

  /** Return top similar items based on items user recently has action on */
  def predictSimilar(
    recentFeatures: Set[Array[Double]],
    productModels: Map[Int, ProductModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = productModels.par // convert to parallel collection
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
    // can add other custom filtering here
    whiteList.map(_.contains(i)).getOrElse(true) &&
    !blackList.contains(i) &&
    // filter categories
    categories.map { cat =>
      item.categories.toList.headOption.map { itemCat =>
        // keep this item if has overlap categories with the query
        !(itemCat.toSet.intersect(cat.toSet[Any]).isEmpty)
      }.getOrElse(false) // discard this item if it has no categories
    }.getOrElse(true)

  }




  /* ContentBasedFiltering */
  private def encode(data: RDD[Array[String]]): RDD[Vector] = {
    val dict = BiMap.stringLong(data.flatMap(x => x))
    val len = dict.size

    data.map { sample =>
      val indexes = sample.map(dict(_).toInt).sorted
      Vectors.sparse(len, indexes, Array.fill[Double](indexes.length)(1.0))
    }
  }

  // [X: ClassTag] - trick to have multiple definitions of encode, they both
  // for RDD[_]
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