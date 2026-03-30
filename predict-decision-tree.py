#pract7 Predict Real Estate Values with Decision Trees in Spark
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# Load the real estate dataset
df = spark.table("workspace.default.real_estate")

# Inspect the dataset schema
df.printSchema()

# Assemble feature columns into a single feature vector
assembler = VectorAssembler(inputCols=["feature1", "feature2"],
outputCol="features")
df = assembler.transform(df)

# Initialize Decision Tree Regressor
dt = DecisionTreeRegressor(labelCol="price", featuresCol="features")

# Train the model
model = dt.fit(df)

# Make predictions
predictions = model.transform(df)

# Show some predictions
predictions.select("features", "price", "prediction").show(5)

# Evaluate the model with RMSE metric
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction",
metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
