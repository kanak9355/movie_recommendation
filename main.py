import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_excel("data/Movie_Recommendation_System.xlsx",sheet_name="Movie_Data")



# Show original shape
print("Original shape:", df.shape)

# Fill missing Genre and Language
df["Genre"] = df["Genre"].fillna("Unknown")
df["Language"] = df["Language"].fillna("Unknown")

# Remove rows where Rating is missing
df = df.dropna(subset=["Rating"])

# Remove duplicate User-Movie records
df = df.drop_duplicates(subset=["User_ID", "Movie_Title"])

print("Cleaned shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

#Convering a table to id and movie format- Matrix

user_movie_matrix = df.pivot_table(
    index="User_ID",
    columns="Movie_Title",
    values="Rating",
    aggfunc="mean"
)

print("\nUser-Movie Matrix:")
print(user_movie_matrix.head())

#For cosine sim fn. we convert NaN val to 0
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# 5. Calculate similarity between users-it will be in numpy format
user_similarity = cosine_similarity(
    user_movie_matrix_filled
)

print(
    "\nUser similarity matrix shape:",
    user_similarity.shape
)

user_id = int(input("Enter User ID: "))

if user_id not in user_movie_matrix.index:
    print("Invalid User ID")
else:
    similarity_df = pd.DataFrame(       # 6. Convert similarity matrix into DataFrame
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

    similar_users = similarity_df[user_id].sort_values(
        ascending=False
    )

    similar_users = similar_users.drop(user_id)

    print("\nMost Similar Users:")
    print(similar_users.head(5))

# Get movies already rated by the selected user
user_movies = user_movie_matrix.loc[user_id]

# Remove movies that the user has not rated
user_movies = user_movies.dropna()

print("\nMovies already rated by User", user_id)
print(user_movies)

# Get the IDs of top 5 similar users
top_users = similar_users.head(5).index

# Get ratings given by these users
similar_users_ratings = user_movie_matrix.loc[top_users]

print("\nRatings of similar users:")
print(similar_users_ratings)

# Calculate average rating for each movie
movie_scores = similar_users_ratings.mean(axis=0)

    # Remove movies already rated by selected user
movie_scores = movie_scores.drop(
        user_movies.index,
        errors="ignore"
    )

    # Sort movies by rating
movie_scores = movie_scores.sort_values(
        ascending=False
    )

    # Get top 5 recommendations
recommendations = movie_scores.head(5)

print("\nRecommended Movies:")
print(recommendations)