package com.recipe

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(
  user: String,
  num: Int,
  categories: Option[Set[String]],
  whiteList: Option[Set[String]],
  blackList: Option[Set[String]]
) extends Serializable

case class PredictedResult(
  itemScores: Array[ItemScore]
) extends Serializable

// case class ItemScore(
//   item: String,
//   score: Double
// ) extends Serializable

case class ItemScore(item: String, score: Double) extends Serializable with
Ordered[ItemScore] {
  def compare(that: ItemScore) = this.score.compare(that.score)
}

object RecommendationEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("CollaborativeAlgorithm" -> classOf[CollaborativeAlgorithm],
        "ContentBasedAlgorithm" -> classOf[ContentBasedAlgorithm]), // ADDED
      classOf[Serving])
  }
}