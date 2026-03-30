# Input file path in DBFS or Workspace
input_file = "/Workspace/Users/atharvabandre@gmail.com/1st.txt"
# Read the text file as a DataFrame
df = spark.read.text(input_file)
from pyspark.sql import functions as F
# Split lines into words, explode, group, and count
word_counts = (
df.select(F.explode(F.split(F.col("value"), r"\s+")).alias("word"))
.filter(F.col("word") != "") # remove empty tokens
.groupBy("word")
.count()
.orderBy(F.desc("count"))
)
# Show results
word_counts.show(truncate=False)
