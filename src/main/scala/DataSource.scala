package com.recipe

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.Event
import io.prediction.data.store.PEventStore

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

case class DataSourceParams(appName: String) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {

    // create a RDD of (entityID, User)
    val usersRDD: RDD[(String, User)] = PEventStore.aggregateProperties(
      appName = dsp.appName,
      entityType = "user"
    )(sc).map { case (entityId, properties) =>
      val user = try {
        User()
      } catch {
        case e: Exception => {
          logger.error(s"Failed to get properties ${properties} of" +
            s" user ${entityId}. Exception: ${e}.")
          throw e
        }
      }
      (entityId, user)
    }.cache()

    // create a RDD of (entityID, Item)
    val itemsRDD: RDD[(String, Item)] = PEventStore.aggregateProperties(
      appName = dsp.appName,
      entityType = "item"
    )(sc).map { case (entityId, properties) =>
      val item = try {
        val title: String = properties.get[String]("title")
        val categories: Array[String] = properties.get[Array[String]]("categories")
        val feelings: Array[String] = properties.get[Array[String]]("feelings")
        val cooktime: Int = properties.get[Int]("cooktime")
        val amount: Int = properties.get[Int]("amount")
        val expire: Int = properties.get[Int]("expire")
        
        // val title: String = properties.get[String]("title")
        // val producer: String = properties.get[String]("producer")
        // val director: String = properties.get[String]("director")
        // val genres: Array[String] = properties.get[Array[String]]("genres")
        // val actors: Array[String] = properties.get[Array[String]]("actors")
        // val year: Int = properties.get[Int]("year")
        // val duration: Int = properties.get[Int]("duration")


        // Assume categories is optional property of item.
        Item(entityId, title, categories, feelings, cooktime, amount, expire)
        //Item(categories, )
      } catch {
        case e: Exception => {
          logger.error(s"Failed to get properties ${properties} of" +
            s" item ${entityId}. Exception: ${e}.")
          throw e
        }
      }
      (entityId, item)
    }.cache()

    val eventsRDD: RDD[Event] = PEventStore.find(
      appName = dsp.appName,
      entityType = Some("user"),
      eventNames = Some(List("view", "like", "cancel_like")), // MODIFIED
      // targetEntityType is optional field of an event.
      targetEntityType = Some(Some("item")))(sc)
      .cache()

    val viewEventsRDD: RDD[ViewEvent] = eventsRDD
      .filter { event => event.event == "view" }
      .map { event =>
        try {
          ViewEvent(
            user = event.entityId,
            item = event.targetEntityId.get,
            t = event.eventTime.getMillis
          )
        } catch {
          case e: Exception =>
            logger.error(s"Cannot convert ${event} to ViewEvent." +
              s" Exception: ${e}.")
            throw e
        }
      }

    val likeEventsRDD: RDD[LikeEvent] = eventsRDD
      .filter { event => event.event == "like" | event.event == "cancel_like"}  // MODIFIED
      .map { event =>
        try {
          LikeEvent(
            user = event.entityId,
            item = event.targetEntityId.get,
            t = event.eventTime.getMillis,
            like = (event.event == "like")
          )
        } catch {
          case e: Exception =>
            logger.error(s"Cannot convert ${event} to LikeEvent." +
              s" Exception: ${e}.")
            throw e
        }
      }

    new TrainingData(
      users = usersRDD,
      items = itemsRDD,
      viewEvents = viewEventsRDD,
      likeEvents = likeEventsRDD  // MODIFIED
    )
  }
}

case class User()

//case class Item(categories: Option[List[String]])

case class Item(item: String, title: String, categories: Array[String], feelings: Array[String], cooktime: Int,
                    amount: Int, expire: Int)

case class ViewEvent(user: String, item: String, t: Long)

case class LikeEvent( // MODIFIED
  user: String,
  item: String,
  t: Long,
  like: Boolean // true: like. false: cancel_like
)

class TrainingData(
  val users: RDD[(String, User)],
  val items: RDD[(String, Item)],
  val viewEvents: RDD[ViewEvent],
  val likeEvents: RDD[LikeEvent] // MODIFIED
) extends Serializable {
  override def toString = {
    s"users: [${users.count()} (${users.take(2).toList}...)]" +
    s"items: [${items.count()} (${items.take(2).toList}...)]" +
    s"viewEvents: [${viewEvents.count()}] (${viewEvents.take(2).toList}...)" +
    s"likeEvents: [${likeEvents.count()}] (${likeEvents.take(2).toList}...)"
  }
}