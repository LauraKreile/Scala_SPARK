package final_work

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object TradingStockVolume extends App {

  val spark = SparkSession.builder().appName("test").master("local").getOrCreate()
  println(s"Session started on Spark version ${spark.version}")

  val stockTic = spark.read.format("csv")
    .option("header", "true")
    .load("./src/resources/DailyStockReturn_ticker.csv")
  stockTic.printSchema()
  stockTic.show(10, false)

  // The def was created to add a short companies description to stock symbols
  def tickerMatch(x: String): String = x match {
    case "AAPL" => "Apple Inc. designs, manufactures and markets mobile communication, media devices, personal computers, portable digital music players, software, networking solutions and applications."
    case "BLK" => "BlackRock, Inc. (BlackRock) is an investment management company. The Company provides a range of investment and risk management services to institutional and retail clients worldwide."
    case "GOOG" => "Alphabet Inc. is a holding company. The Company's businesses include Google Inc. (Google) and its Internet products, such as Access, Calico, CapitalG, GV, Nest, Verily, Waymo and X."
    case "MSFT" => "Microsoft Corporation is a technology company. The Company develops, licenses, and supports a range of software products, services, devices, operating systems, applications etc."
    case "TSLA" => "Tesla, Inc. designs, develops, manufactures and sells electric vehicles and designs, manufactures installs and sells solar energy generation and energy storage products."
    case _ => "not value"
  }

  val addDescriptionUDF = udf((i: String) => {
    tickerMatch(i)
  })

  // The new column 'Company description' was added and the DF data types were casted
  val stockTicDescription = stockTic
    .withColumn("Company_description", addDescriptionUDF(col("ticker")))
    .withColumn("date", col("date").cast("date"))
    .withColumn("close", col("close").cast("double"))
    .withColumn("daily_return", col("daily_return").cast("double"))
    .withColumn("average_daily_return", col("average_daily_return").cast("double"))
    .withColumn("max_daily_return", col("max_daily_return").cast("double"))
    .withColumn("min_daily_return", col("min_daily_return").cast("double"))
    .withColumn("daily_return_CumSal", col("daily_return_CumSal").cast("double"))
    .withColumn("volume", col("volume").cast("integer"))
  stockTicDescription.show(5, false)

  // The most active trading volumes are:
  val windowSpec = Window
    .partitionBy("ticker") // it will be in a specific on column 'ticker' (GOOG, AAPL, BLK, MSFT, TSLA)
    .orderBy(col("volume").desc)
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

  val avgVolume = avg(col("volume")).over(windowSpec)
  val maxVolume = max(col("volume")).over(windowSpec)
  val minVolume = min(col("volume")).over(windowSpec)
  val sumVolume = sum(col("volume")).over(windowSpec)

  // 1. The most active average stock trading volume of each stock for each day
  val stockVolumeEveryDay = stockTicDescription
    .select(
      col("date"),
      col("open"),
      col("close"),
      col("daily_return"),
      col("average_daily_return"),
      col("volume"),
      avgVolume.alias("average_volume"),
      sumVolume.alias("sum_volume"),
      maxVolume.alias("max_volume"),
      minVolume.alias("min_volume"),
      col("ticker"),
      col("Company_description"))
    .withColumn("average_volume", col("average_volume").cast("long"))
    .orderBy(col("average_volume").desc)
  stockVolumeEveryDay.show(10, false)
  stockVolumeEveryDay.summary().show() // here we can see max, min of 'average volume' as well
  stockVolumeEveryDay.printSchema()


  // 2. The most active average stock trading volume of all stocks for each day
  val allStockVolumeDay = stockTicDescription
    .groupBy("date")
    .agg("volume" -> "sum", "volume" -> "mean"
      , "daily_return" -> "mean")
    .orderBy(desc("avg(volume)"))
    .withColumn("sum(volume)", col("sum(volume)").cast("long"))
    .withColumn("avg(volume)", col("avg(volume)").cast("long"))
  allStockVolumeDay.show(10)
  allStockVolumeDay.printSchema()

  // 3.The most active average stock trading volume of all stocks for each month
  val allStockVolumeMonth = stockTicDescription
    .groupBy(month(col("date")).alias("month"))
    .agg("volume" -> "sum", "volume" -> "mean"
      , "daily_return" -> "mean", "daily_return" -> "max")
    .orderBy(desc("avg(volume)"))
    .withColumn("sum(volume)", col("sum(volume)").cast("long"))
    .withColumn("avg(volume)", col("avg(volume)").cast("long"))
    .show(5)

  // 4.The most active average stock trading volume of each stock of each quarter
  val stockTicVolumeQuarter = stockTicDescription
    .groupBy(quarter(col("date")).alias("quarter"), col("ticker"))
    .agg("volume" -> "sum", "volume" -> "mean"
      , "volume" -> "max", "volume" -> "min"
      , "daily_return" -> "mean", "daily_return" -> "max")
    .withColumn("avg(volume)", col("avg(volume)").cast("long"))
    .orderBy(asc("quarter"), desc("avg(volume)"))
  stockTicVolumeQuarter.show()


  /** The most frequently traded stocks are: **/
  // The frequency was calculated by the following formula (- as measured by closing price * volume - on average)
  // 1.The most frequently traded stock for each day
  val frequentlyTradedStock = stockVolumeEveryDay
    .withColumn("trading_frequency",
      col("close") * col("volume") - col("average_volume"))
    .select("date", "open", "close", "daily_return", "average_daily_return"
      , "volume", "average_volume", "trading_frequency", "ticker", "Company_description")
    .withColumn("trading_frequency", col("trading_frequency").cast("long"))
    .orderBy(desc("trading_frequency"))
  frequentlyTradedStock.show(10, false)
  frequentlyTradedStock.printSchema()
  frequentlyTradedStock.summary().show()

  // 2.The most frequently traded stock for each quarter
  val frequentlyTradedStockQuarter = frequentlyTradedStock
    .groupBy(quarter(col("date")).alias("quarter"), col("ticker"))
    .agg("trading_frequency" -> "mean", "trading_frequency" -> "sum"
      , "trading_frequency" -> "max", "trading_frequency" -> "min", "daily_return" -> "mean")
    .withColumn("avg(trading_frequency)", col("avg(trading_frequency)").cast("long"))
    .orderBy(asc("quarter"), desc("avg(trading_frequency)"))
  frequentlyTradedStockQuarter.show()

  // 3.The most frequently traded stock in year
  val frequentlyTradedStockYear = frequentlyTradedStock
    .groupBy(year(col("date")).alias("year"), col("ticker"))
    .agg("trading_frequency" -> "mean", "trading_frequency" -> "sum"
      , "trading_frequency" -> "max", "trading_frequency" -> "min", "daily_return" -> "mean")
    .withColumn("avg(trading_frequency)", col("avg(trading_frequency)").cast("long"))
    .orderBy(asc("year"), desc("avg(trading_frequency)"))
  frequentlyTradedStockYear.show()

}
