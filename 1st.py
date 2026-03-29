'''Big data is powerful
Big data is scalable
Spark makes big data processing fast
'''
#1 Word Count/Frequency using MapReduce
from pyspark.sql.functions import split, explode
df = spark.read.text("Add path of text file")
counts = (
    df.select(explode(split(df["value"], " ")).alias("word"))
      .groupBy("word")
      .count()
)
counts.show()
