#decision tree
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Load data
df = spark.table("workspace.default.iris")
# StringIndexer for label column
indexer = StringIndexer(inputCol="Species", outputCol="label")
df_indexed = indexer.fit(df).transform(df)
# Assemble features
feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm",
"PetalWidthCm"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_features = assembler.transform(df_indexed)
# Split data
train_df, test_df = df_features.randomSplit([0.7, 0.3], seed=42)

# Decision Tree model
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
model = dt.fit(train_df)

# Predict
predictions = model.transform(test_df)

# Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="label",
predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

display(predictions.select("Id", "Species", "prediction"))
