package com.recipe

import io.prediction.controller.LServing

import grizzled.slf4j.Logger

import breeze.stats.mean
import breeze.stats.meanAndVariance
import breeze.stats.MeanAndVariance

class Serving
  extends LServing[Query, PredictedResult] {

  override
  def serve(query: Query,
    predictedResults: Seq[PredictedResult]): PredictedResult = {

    @transient lazy val logger = Logger[this.type]

    // 같은 item에 대해 Collaborative Filtering Recommender의 결과값과 
    // Content Based Filtering Recommender의 결과값을 합산, 정렬함
    val combined = predictedResults.map(_.itemScores).flatten // ItemScore 배열
      .groupBy(_.item) // 같은 item id끼리 묶음
      .mapValues(itemScores => itemScores.map(_.score).reduce(_ + _))
      .toArray // (item id, score) 배열
      .sortBy(_._2)(Ordering.Double.reverse)
      .take(query.num)
      .map { case (k,v) => ItemScore(k, v) }
    
    if (!combined.isEmpty) logger.info(s"Recommendation result for user ${query.user} is successfully sent to the user.")

    new PredictedResult(combined)
  }
}