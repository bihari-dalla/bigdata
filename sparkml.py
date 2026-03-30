# Train and evaluate using Spark ML to Produce wine Recommendations
# on wine.csv datasets

# ============================================
# 1. Load Required Libraries
# ============================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import numpy as np

# ============================================
# 2. Load wine.csv from DBFS
# ============================================
df = spark.table("default.wine")

# Rename columns with dots to valid identifiers
# Only rename columns with dots: Malic.acid, Nonflavanoid.phenols, Color.int
df = df.withColumnRenamed("Malic.acid", "Malic_acid") \
       .withColumnRenamed("Nonflavanoid.phenols", "Nonflavanoid_phenols") \
       .withColumnRenamed("Color.int", "Color_int")

# Add unique wine ID
df = df.withColumn("wine_id", monotonically_increasing_id())

print("Dataset Schema:")
df.printSchema()
df.show(5)

# ============================================
# 3. Select Feature Columns (Numerical Only)
# ============================================
feature_cols = [c for c in df.columns if c not in ["wine_id"]]

# ============================================
# 4. Assemble Features
# ============================================
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df_features = assembler.transform(df)

# ============================================
# 5. Scale Features
# ============================================
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)
