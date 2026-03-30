#Practical 5 Implement Spark program to Movie Lens Movie Rating Dataset. Run your first Spark program! Ratings histogram example.
# Load the ratings dataset
ratings_df = spark.table("workspace.default.movie_ratings")

# Group by rating and count occurrences
histogram_df = ratings_df.groupBy("rating").count().orderBy("rating")

# Display the histogram as a bar chart
display(histogram_df)  # In Databricks, use the visualization options above the table to select "Bar chart" for a histogram view.
