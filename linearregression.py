#linearregression ,fabricate, online store data
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import numpy as np

# Load data
df = spark.table("default.revenue_speed_linear_regression")

# Prepare features
assembler = VectorAssembler(
    inputCols=["speed"],
    outputCol="features"
)
df_features = assembler.transform(df)

# Fit linear regression model
lr = LinearRegression(
    featuresCol="features",
    labelCol="revenue"
)
lr_model = lr.fit(df_features)

# Generate fabricated data for average page speed
avg_speeds = np.linspace(
    df.selectExpr("min(speed)").first()[0],
    df.selectExpr("max(speed)").first()[0],
    10
)

fabricated_df = spark.createDataFrame(
    [(float(s),) for s in avg_speeds],
    ["speed"]
)

fabricated_df_features = assembler.transform(fabricated_df)

# Predict revenue for fabricated speeds
predictions = lr_model.transform(fabricated_df_features).select(
    "speed",
    "prediction"
)

display(predictions)
