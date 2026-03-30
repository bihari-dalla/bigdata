#pract8 Build a Movie Rating Prediction System using Random Forest in PySpark

from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

# Initialize Spark session
spark = SparkSession.builder.appName("MovieRatingPrediction").getOrCreate()

# Load the dataset
df = spark.table("workspace.default.ratings_small")

# Inspect the dataset
df.printSchema()

# Selecting required columns
df = df.select("userId", "movieId", "rating")

# Convert features into a single vector column
feature_assembler = VectorAssembler(inputCols=["userId", "movieId"], outputCol="features")
df = feature_assembler.transform(df).select("features", "rating")

# Split into training and testing datasets
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

# Initialize Random Forest model
rf = RandomForestRegressor(featuresCol="features", labelCol="rating", numTrees=100)

# Train the model
model = rf.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate using RMSE
evaluator = RegressionEvaluator(
    labelCol="rating",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Show predictions
predictions.select("features", "rating", "prediction").show(10, truncate=False)
