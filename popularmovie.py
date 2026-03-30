#pract6 Implement Spark program to Find the Most Popular Movie
movies = spark.read.table(
"workspace.default.movie_ratings"
)
popular_movies = (
movies
.groupBy("movieId")
.count()
.orderBy("count", ascending=False)
)
display(popular_movies.limit(1))
