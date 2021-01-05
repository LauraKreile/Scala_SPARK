package final_work

import java.io.File

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object DailyStockReturn extends App {

  val spark = SparkSession.builder().appName("test").master("local").getOrCreate()
  println(s"Session started on Spark version ${spark.version}")

  val fPath = "./src/resources/stock_prices.csv"
  val df = spark.read
    .format("csv")
    .option("header", true)
    .load(fPath)

  println(df.summary().show())
  df.printSchema()
  df.show(10, false)

  // The DF data types were casted
  val df2 = df
    .withColumn("date", col("date").cast("date"))
    .withColumn("open", col("open").cast("double"))
    .withColumn("high", col("high").cast("double"))
    .withColumn("low", col("low").cast("double"))
    .withColumn("close", col("close").cast("double"))
    .withColumn("volume", col("volume").cast("integer"))
  df2.printSchema()
  df2.show(5, false)

  // The daily return was calculated from 'close' and 'open" prices
  val df3 = df2
    .withColumn("daily_return", (df2.col("close") - df2.col("open")) / df2.col("open") * 100) //a percentage
    .withColumn("percentage_rounded", round(col("daily_return"), 2))
  df3.select("daily_return", "percentage_rounded").show(5, false)
  df3.summary().show() //here we can check 'daily_stock_return' max value, min value and mean value
  df3.printSchema()

  // Analysis of daily stock return
  // 1.The highest daily return
  df3.select(col("date"),
    col("open"),
    col("close"),
    col("daily_return"),
    col("percentage_rounded"),
    col("volume"),
    col("ticker"))
    .orderBy(col("daily_return").desc)
    .show(10, false)

  df3.printSchema()

  // 2.The lowest daily return with stock ticker
  val lowestDailyReturn = df3.select("date", "daily_return", "ticker")
    .orderBy(col("daily_return").asc)
    .show(1, false)

  // A window function was created to get avg daily returns for each day
  val windowSpec = Window
    .partitionBy("ticker") // it will be in a specific on column 'ticker' (GOOG, AAPL, BLK, MSFT, TSLA)
    .orderBy(col("daily_return").desc)
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

  val windowSpec2 = Window.partitionBy("ticker")
    .orderBy(col("date")).rowsBetween(Window.unboundedPreceding, Window.currentRow) // to get 'running total'

  val avgVolume = avg(col("daily_return")).over(windowSpec)
  val maxVolume = max(col("daily_return")).over(windowSpec)
  val minVolume = min(col("daily_return")).over(windowSpec) //min is giving the current
  val sumVolume = sum(col("daily_return")).over(windowSpec2) //for running sum

  // 3. The daily returns (agg) for each stock ticker per day
  val dailyReturnTickerDay = df3  /** this result will be save to file **/
    .select(
      col("date"),
      col("open"),
      col("close"),
      col("daily_return"),
      avgVolume.alias("average_daily_return"),
      maxVolume.alias("max_daily_return"),
      minVolume.alias("min_daily_return"), //min is giving the current
      sumVolume.alias("daily_return_CumSal"),
      col("volume"), //running sum from column 'daily_stock_return'
      col("ticker"))
    .orderBy(col("date").desc)

  dailyReturnTickerDay.show(10, false)
  dailyReturnTickerDay.printSchema()

  // To check cumulative sum (Running total) of daily stock return
  dailyReturnTickerDay
    .where(col("ticker").contains("GOOG"))
    .orderBy(asc("daily_return_CumSal"))
    .show(5)

  // 4.The daily stock return of all stocks for every day
  val allStockReturnDay = df3  /** this result will be save to file **/
    .groupBy("date")
    .agg("daily_return" -> "mean", "daily_return" -> "max",
      "daily_return" -> "min", "daily_return" -> "sum")
    .withColumnRenamed("avg(daily_return)", "average_daily_return") // The brackets were removed, because the result couldn't be saved to a file
    .withColumnRenamed("max(daily_return)", "max_daily_return")
    .withColumnRenamed("min(daily_return)", "min_daily_return")
    .withColumnRenamed("sum(daily_return)", "sum_daily_return")
    .orderBy(desc("average_daily_return"))
  allStockReturnDay.show(10, false)

  // 5.The daily stock return of each stocks ticker for each year
  val dailyReturnTicYear = df3
    .groupBy(year(col("date")).alias("year"), col("ticker"))
    .agg("daily_return" -> "mean", "daily_return" -> "max"
      , "daily_return" -> "min", "daily_return" -> "sum")
    .orderBy(asc("year"), desc("avg(daily_return)"))
  dailyReturnTicYear.show()

  // 5.The daily stock return of each stocks ticker for each quarter
  val dailyReturnTicQuarter = df3
    .groupBy(quarter(col("date")).alias("quarter"), col("ticker"))
    .agg("daily_return" -> "mean", "daily_return" -> "max",
      "daily_return" -> "min", "daily_return" -> "sum")
    //.withColumnRenamed("avg(daily_return)", "average_return")
    .orderBy(asc("quarter"), desc("avg(daily_return)"))
    .show()

  // 6. The daily stock return of each stock ticker in total
  val dailyReturnTicAll = df3.groupBy("ticker")
    .agg(mean("daily_return"), max("daily_return")
      , min("daily_return"), mean("open"), mean("close"))
    .orderBy(desc("max(daily_return)")).show()

  // 7. The daily stock return of each stock for year, using pivot
  val pivotedAggYear = df3
    .groupBy(year(col("date")))
    .pivot("ticker")
    .agg("open" -> "mean", "close" -> "mean"
      , "daily_return" -> "mean", "daily_return" -> "sum"
      , "volume" -> "sum", "volume" -> "mean")
  pivotedAggYear.show()

  //Save to Parquet format
  val fPath20 = "./src/resources/Assignment.parquet"
  allStockReturnDay
    .coalesce(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(fPath20)

  val newPath20 = "./src/resources/DailyStockReturn.parquet" //new path to single file

  val dir = new File(fPath20)
  dir.listFiles.foreach(println)
  //this will delete file map and will create new file, renaming build file (will be single file)
  val tmpTsfFile = dir.listFiles.filter(_.toPath.toString.endsWith(".parquet"))(0).toString
  new File(tmpTsfFile).renameTo(new File(newPath20))

  dir.listFiles.foreach(f => f.delete) //delete all the files in the directory 'SparkAssignment_stock.parquet'
  dir.delete //delete itself

  //Read Parquet file
  val assignmentStock = spark.read
    .format("parquet")
    .load(newPath20)
  assignmentStock.show(5, false)
  assignmentStock.printSchema()

  //Save to Json format
  val fPath202 = "./src/resources/Assignment_stock.json"
  allStockReturnDay
    .coalesce(1)
    .write
    .format("json")
    .mode("overwrite")
    .save(fPath202)

  val newPath202 = "./src/resources/DailyStockReturn.json" //new path to single file

  val dir2 = new File(fPath202)
  dir2.listFiles.foreach(println)

  val tmpTsfFile2 = dir2.listFiles.filter(_.toPath.toString.endsWith(".json"))(0).toString
  new File(tmpTsfFile2).renameTo(new File(newPath202))

  dir2.listFiles.foreach(f => f.delete)
  dir2.delete

  //Read Json file
  val assignmentStock2 = spark.read
    .format("json")
    .load(newPath202)
  assignmentStock2.show(5, false)
  assignmentStock2.printSchema()


  //Save csv
  val fPath2020 = "./src/resources/Assignment_stock.csv"
  dailyReturnTickerDay
    .coalesce(1)
    .write
    .option("header", "true") //added header
    .format("csv")
    .mode("overwrite")
    .save(fPath2020)

  val newPath2020 = "./src/resources/DailyStockReturn_ticker.csv" //new path to single file

  val dir3 = new File(fPath2020)
  dir3.listFiles.foreach(println)

  val tmpTsfFile3 = dir3.listFiles.filter(_.toPath.toString.endsWith(".csv"))(0).toString
  new File(tmpTsfFile3).renameTo(new File(newPath2020))

  dir3.listFiles.foreach(f => f.delete)
  dir3.delete

  //Read
  val assignmentStockTicker = spark.read
    .format("csv")
    .option("header", true) //will use first row for header
    .load(newPath2020)
  assignmentStockTicker.show(5, false)
  assignmentStockTicker.printSchema()


}
