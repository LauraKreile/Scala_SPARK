package final_work

import org.apache.spark.ml.feature.{ RFormula, StandardScaler }
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{ col, desc, lit, round }

object v_ML_DailyStockReturn extends App {

  val spark = SparkSession.builder().appName("test").master("local").getOrCreate()
  println(s"Session started on Spark version ${spark.version}")

  val stockTic = spark.read.format("csv")
    .option("header", "true")
    .load("./src/resources/DailyStockReturn_ticker.csv")

  stockTic.show(4, false)
  stockTic.printSchema()

  //The DF 'stockTic' data types were casted
  val stockTicCasted = stockTic
    .withColumn("daily_return", col("daily_return").cast("double"))
    .withColumn("average_daily_return", col("average_daily_return").cast("double"))
    .withColumn("max_daily_return", col("max_daily_return").cast("double"))
    .withColumn("min_daily_return", col("min_daily_return").cast("double"))
    .withColumn("open", col("open").cast("double"))
    .withColumn("close", col("close").cast("double"))
    .orderBy(desc("date"))

  stockTicCasted.show(5)
  stockTicCasted.printSchema()

  // Predictions of daily stock return
  // The label and features were created from daily stock returns data
  val supervised = new RFormula()
    .setFormula("daily_return ~ . + ticker:average_daily_return + ticker:max_daily_return " +
      " + ticker:min_daily_return")

  val fittedRF = supervised.fit(stockTicCasted)
  val preparedDF = fittedRF.transform(stockTicCasted)
  preparedDF.show(5, false)

  val output = supervised.fit(stockTicCasted).transform(stockTicCasted) //the same results
  println("The label and features from daily stock returns")
  output.select("features", "label").show(10, true)

  val lr = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
  val lrModel = lr.fit(output)
  val predictionsDailyReturn = lrModel.transform(output)
  predictionsDailyReturn.show(5, false)
  println("Predictions of daily stock return")
  predictionsDailyReturn
    .select("date", "ticker", "features", "label", "prediction")
    .show(10, true)

  // The model summary with metrics
  val modelSummary = lrModel.summary
  println(s"RMSE: ${modelSummary.rootMeanSquaredError}") // RMSE: 0.30095784983770446. The results are very good! (<0.5 and <0.3, respectively)
  println(s"r2: ${modelSummary.r2}")
  // Without these parameters = setMaxIter(10) .setRegParam(0.3).setElasticNetParam(0.8), the result
  // of Root Mean Square Error wasn't good. The residuals were a lot of far from the regression line data points.

  // Residuals were checked
  modelSummary.residuals.show()

  // The data set of "functions" were transformed into a standardization dataset // Before that, the RMSE result was worst, so a standard scale was created
  val transformedScaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("features_scaled")
  val outputScl = transformedScaler.fit(output).transform(output)
    .withColumnRenamed("features", "old_features")
    .withColumnRenamed("features_scaled", "features")
  outputScl.select("old_features", "features").show(5, false)

  val lrModelScl = lr.fit(outputScl)
  val predictionsDailyReturnScl = lrModelScl.transform(outputScl)
  predictionsDailyReturn.show(5, false)
  println("Predictions of daily stock return with features scaled")
  predictionsDailyReturn
    .select("date", "ticker", "features", "label", "prediction")
    .show(10, true)

  // The model summary with scaled features
  // The same results, although the features are slightly different
  val modelSclSummary = lrModelScl.summary
  println(s"RMSE: ${modelSclSummary.rootMeanSquaredError}") // RMSE: 0.30095784983770446
  println(s"r2: ${modelSclSummary.r2}")


  // Data were divided into training and test sets (30% were kept for testing)
  val Array(train, test) = output.randomSplit(Array(0.7, 0.3))

  val lr1 = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
  val modelTest = lr1.fit(train)
  val predictDailyReturnTest = modelTest.transform(test)
  println("Predictions of daily stock return using data split with train, test")
  predictDailyReturnTest
    .select("date", "ticker", "features", "label", "prediction")
    .show(5, true)

  // The model summary with metrics
  val modelTestSummary = modelTest.summary
  println(s"RMSE: ${modelTestSummary.rootMeanSquaredError}")
  println(s"r2: ${modelTestSummary.r2}")
  // Now the results are better, RMSE is lower and r2 is slightly higher
  // Before were RMSE: 0.30095784983770446, r2: 0.9537144170731122
  // Now are RMSE 0.2956098469242731, r2: 0.9545914173556882

  // Predictions of the price of each stock
  val fPath = "./src/resources/stock_prices.csv"
  val df = spark.read
    .format("csv")
    .option("header", true)
    .load(fPath)
  df.show(5, false)
  df.printSchema()

  // The DF data types were casted
  val casDf = df
    .withColumn("open", col("open").cast("double"))
    .withColumn("high", col("high").cast("double"))
    .withColumn("low", col("low").cast("double"))
    .withColumn("close", col("close").cast("double"))
    .orderBy(desc("date"))
  casDf.show(10, false)

  // The label and features were created with data of each ticker prices
  val supervised2 = new RFormula()
    .setFormula("open ~ . + ticker:high + ticker:low + ticker:close")

  val output2 = supervised2.fit(casDf).transform(casDf)

  val lr3 = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.1)
    .setElasticNetParam(0.1) // The next models will require these instance parameters
  val linearModel2 = lr3.fit(output2)
  val predictionsPrice = linearModel2.transform(output2)
  predictionsPrice.show(5, false)
  println("Predictions of price")
  predictionsPrice
    .select("date", "ticker", "features", "label", "prediction")
    .show(10, true)

  // The model summary with metrics
  val modelSummary1 = linearModel2.summary
  println(s"RMSE: ${modelSummary1.rootMeanSquaredError}") // RMSE: 0.33901802211601473. The results are very good! (<0.5 and <0.3, respectively)
  println(s"r2: ${modelSummary1.r2}")

  // The data set of "functions" were transformed into a standardization dataset
  val transformedScaler1 = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("features_scaled")
  val output2Scl = transformedScaler1.fit(output2).transform(output2)
    .withColumnRenamed("features", "old_features")
    .withColumnRenamed("features_scaled", "features")
  output2Scl.select("old_features", "features").show(5, false)

  val linearSclModel2 = lr3.fit(output2Scl)
  val predictionsPriceScl = linearSclModel2.transform(output2Scl)
  predictionsPriceScl.show(5, false)
  println("Predictions of daily stock return with features scaled")
  predictionsPriceScl
    .select("date", "ticker", "features", "label", "prediction")
    .show(10, true)

  // The model summary with metrics
  // The same results, although the features are slightly different
  val modelSummaryScl1 = linearSclModel2.summary
  println(s"RMSE: ${modelSummaryScl1.rootMeanSquaredError}")
  println(s"r2: ${modelSummaryScl1.r2}")

  // Data were divided into training and test sets (30% were kept for testing)
  val Array(train1, test1) = output2.randomSplit(Array(0.7, 0.3), seed = 2020)
  val modelTest1 = lr3.fit(train1)
  val predictPriseTest = modelTest1.transform(test1)
  println("Predictions of stock prices using data split with train, test")
  predictPriseTest.show(5, true)

  // The model summary with metrics
  val modelTest1Summary = modelTest1.summary
  println(s"RMSE: ${modelTest1Summary.rootMeanSquaredError}")
  println(s"r2: ${modelTest1Summary.r2}")
  // Now the results are better, RMSE is lower and r2 is slightly higher
  // Before were RMSE: 0.33901802211601473, r2: 0.9999981006246376
  // Now are RMSE 0.23964097688546135, r2: 0.9999990616706848

  import org.apache.spark.ml.feature.VectorAssembler
  // The features were created using Vector
  val va = new VectorAssembler()
    .setInputCols(Array("high", "low", "close"))
    .setOutputCol("features")

  val pricesStock = va.transform(casDf)
    .withColumn("label", lit(col("open"))) // The label was gotten
  println("The labels and features from stock prices")
  pricesStock.show(10, false)

  val lr4 = new LinearRegression()
    .setMaxIter(6)
    .setRegParam(0.1)
    .setElasticNetParam(0.6)
  val lrModel2 = lr4.fit(pricesStock)
  val predictionsPrices = lrModel2.transform(pricesStock)
  println("Predictions of stock price using features from Vector")
  predictionsPrices
    .withColumn("prediction_round", round(col("prediction"), 2))
    .select("date", "ticker", "features", "label", "prediction", "prediction_round")
    .show(10, false)

  // The model summary with metrics
  val trainingSummary = lrModel2.summary
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")
  // RMSE: 2.555393851866969. This result isn't good. This model is not suitable for data with Vector

}
