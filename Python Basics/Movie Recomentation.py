# Movie Recommendation System
# Problem:
# Create a simple movie recommendation system using sets.
# User A’s watched movies (set1)
# User B’s watched movies (set2)
# Program should show:
# Movies both have watched (intersection)
# Movies unique to User A (difference)
# Suggested movies for User A based on User B’s watched list (set2 – set1)

movie_user_a = {"The Godfather","The Dark Knight","12 Angry Men","Fight Club","The Matrix","Interstellar","It's a Wonderful Life","City Lights"}
movie_user_b = {"The Matrix","Interstellar","It's a Wonderful Life","City Lights","Dune","The Shining","12th Fail","Toy Story"}

print("User A :",movie_user_a)
print("User B :",movie_user_b)


print("Movies Watched by Both Users :",movie_user_a & movie_user_b)
print("Movies Unique to User A :",movie_user_a - movie_user_b)
print("Suggested movies for User A :",movie_user_b - movie_user_a)