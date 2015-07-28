package com.recipe

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.Event
import io.prediction.data.storage.Storage
import io.prediction.data.store.PEventStore

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

case class DataSourceEvalParams(kFold: Int, queryNum: Int)

case class DataSourceParams(
  appName: String,
  appId: Int,
  evalParams: Option[DataSourceEvalParams]) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, ActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {
    val eventsDb = Storage.getPEvents()
    val eventsRDD: RDD[Event] = eventsDb.find(
      appId = dsp.appId,
      entityType = Some("user"),
      eventNames = Some(List("view", "like", "cancel_like")), // read "view" and "like" and "cancel_like" event
      // targetEntityType is optional field of an event.
      targetEntityType = Some(Some("item")))(sc)

    val ratingsRDD: RDD[Rating] = eventsRDD.map { event =>
      try {
        val ratingValue: Double = event.event match {
          case "view" => 1.0 // MODIFIED, map view event to rating value of 1
          case "like" => 5.0 // MODIFIED, map like event to rating value of 5
          case "cancel_like" => -5.0 // MODIFIED, map cancel_like event to rating value of -5 
          case _ => throw new Exception(s"Unexpected event ${event} is read.")
        }

        // MODIFIED
        // key is (user id, item id)
        // value is the rating value
        ((event.entityId, event.targetEntityId.get), ratingValue)

      } catch {
        case e: Exception => {
          logger.error(s"Cannot convert ${event} to Rating. Exception: ${e}.")
          throw e
        }
      }
    }
    // MODIFIED
    // sum all values for the same user id and item id key
    .reduceByKey { case (a, b) => a + b }
    .map { case ((uid, iid), r) =>
      Rating(uid, iid, r)
    }.cache()

    new TrainingData(ratingsRDD)
  }
}


case class Rating(
  user: String,
  item: String,
  rating: Double
)

class TrainingData(
  val ratings: RDD[Rating]
) extends Serializable {
  override def toString = {
    s"ratings: [${ratings.count()}] (${ratings.take(2).toList}...)"
  }
}