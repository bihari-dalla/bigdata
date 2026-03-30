#pip install pyspark
#settinng up the sparksession
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("Spark SQL Practical Guide") \
    .getOrCreate()

#creating dataframes
data1 = [
    (1, "Aditi", 29, "Female"),
    (2, "Bharat", 30, "Male"),
    (3, "Chetan", 31, "Male"),
    (4, "Dhruv", 32, "Male")
]

columns1 = ["id", "name", "age", "Gender"]
df1 = spark.createDataFrame(data=data1, schema=columns1)

data2 = [
    (1, "HR"),
    (2, "Sales"),
    (3, "Marketing"),
    (4, "Finance")
]

columns2 = ["id", "department"]
df2 = spark.createDataFrame(data=data2, schema=columns2)

df1.show()
df2.show()

#filtering data
#using where()
df_female = df1.where(df1.Gender == "Female")
df_female.show()

#filtering data
#using filter()
df_above_30 = df1.filter(df1.age > 30)
df_above_30.show()

#adding a column
from pyspark.sql.functions import col
df_new_age = df1.withColumn("age_after_5years", col("age") + 5)
df_new_age.show()

#renaming a column
df_renamed = df1.withColumnRenamed("Gender", "sex")
df_renamed.show()

#dropping a column
df_dropped = df1.drop("age")
df_dropped.show()

#removing duplicate
data_dup = [
    (1, "Atharva", 20, "Male"),
    (2, "Bharat", 30, "Male"),
    (1, "Atharva", 20, "Male")
]
df_dup = spark.createDataFrame(data=data_dup, schema=columns1)

df_distinct = df_dup.distinct()
df_distinct.show()


#grouping data
df_grouped = df1.groupBy("Gender").avg("age")
df_grouped.show()

#joining dataframes
df_joined = df1.join(df2, on="id", how="inner")
df_joined.show()

#RDD TRANSFORMATION METHODS
#USING MAP()
rdd = df1.rdd
rdd_mapped = rdd.map(lambda row: (row.id, row.name,upper()))
print(rdd_mapped.collect())

#using mapPartition()
def to_lower(iterator):
    for row in iterator:
        yield (row.id, row.name.lower())
rdd_partioned = rdd.mapPartitions(to_lower)
print(rdd_partioned.collect())

#element processing methods
#using for foreach()
def print_name(row):
    print(f"Name: {row.name}")
df.foreach(print_name)


#foreachPartition
def print_partition(partition):
    for row in partition:
        print(f"Partitioned Name: {row.name}")
df1.foreachPartition(print_partition)


#pivoting data
data_sales = [
    ("Aditi", 2019, 5000),
    ("Bharat", 2019, 6000),
    ("Chetan", 2020, 7000),
    ("Dhruv", 2020, 8000)
]

columns_sales = ["name", "year", "sales"]
df_sales = spark.createDataFrame(data=data_sales, schema=columns_sales)
df_pivot = df_sales.groupBy("name").pivot("year").sum("sales")
df_pivot.show()


#combining dataframes
data_new = [
    (3, "Esha", 25, "Female")
]
df_new = spark.createDataFrame(data=data_new, schema=columns1)
df_union = df1.union(df_new)
df_union.show()


#collecting data
rows = df1.collect()
for row in rows:
    print(row)

#caching data
df1_cached = df1.cache()
df1_cached.count()

#using persist
df1_persisted = df1_persist(StorageLevel.MEMORY_AND_DISK)
df1_persisted.count()


from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def age_group(age):
    if age < 30:
        return "young"
    elif age < 40:
        return "adult"
    else:
        return "senior"

age_group_udf = udf(age_group, StringType())
df_with_age_group = df1.withColumn("age_group", age_group_udf(df1.age))
df_with_age_group.show()


### SPARK AGGREGATE
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
spark = SparkSession.builder.appName("Spark practical example").getOrCreate()
data = [
    ("Alice","Sales",1000),
    ("Bob","Sales",2000),
    ("Charlie","Marketing",3000),
    ("David","Sales",4000),
    ("Eve","Marketing",5000)
]

df = spark.createDataFrame(data, ["name","department","salary"])
grouped_df = df.groupBy("department").agg(F.sum("salary").alias("total_salary"))
grouped_df.show()

#count distinct values in the department column
distinct_count = df.select(F.countDistinct("department") \
    .alias("distinct_departments"))
distinct_count.show()


#add row number to dataframe
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# Define a window specification, ordering by 'Salary'
window_spec = Window.orderBy("Salary")

# Add a row number column to the DataFrame
df_with_row_number = df.withColumn("Row_Number", row_number().over(window_spec))

# Show the result
df_with_row_number.show()


#select the first row in each group
window_spec = Window.partitionBy("Department").orderBy("Salary")

df_first_row = df.withColumn("Row_Number", row_number().over(window_spec)) \
    .filter(F.col("Row_Number") == 1) \
    .drop("Row_Number")

df_first_row.show()
