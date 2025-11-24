import pandas as pd
import ast
from collections import Counter

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------

movies_path = "tmdb_5000_movies.csv"
credits_path = "tmdb_5000_credits.csv"

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

# Take the "top 500" movies as the first 500 rows
# (adjust this logic if you define "top" differently, e.g. sort by vote_average)
movies_top = movies.sort_values(by="vote_average", ascending=False).head(500).copy()

# ---------------------------------------------------------------------
# 2. Merge with credits on movie id
# ---------------------------------------------------------------------

merged = movies_top.merge(
    credits,
    left_on="id",
    right_on="movie_id",
    how="left"
)

# Fix duplicate title columns (movies + credits both have a title)
if "title_x" in merged.columns:
    merged = merged.rename(columns={"title_x": "title"})
if "title_y" in merged.columns:
    merged = merged.drop(columns=["title_y"])


# ---------------------------------------------------------------------
# 3. Helpers to parse JSON-like string columns (genres, cast)
# ---------------------------------------------------------------------

def parse_list_column(s):
    """
    Safely parse a string representation of a list of dicts.
    Returns [] on errors or NaN.
    """
    if pd.isna(s):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


# Extract list of genre names per movie
merged["genre_names"] = merged["genres"].apply(
    lambda x: [
        d.get("name") for d in parse_list_column(x)
        if isinstance(d, dict) and "name" in d
    ]
)

# Extract list of cast (actor) names per movie
merged["cast_names"] = merged["cast"].apply(
    lambda x: [
        d.get("name") for d in parse_list_column(x)
        if isinstance(d, dict) and "name" in d
    ]
)

# ---------------------------------------------------------------------
# 4. Find top 10 actors and top 10 genres in these 500 movies
# ---------------------------------------------------------------------

actor_counter = Counter(
    name
    for names in merged["cast_names"]
    for name in names
)

genre_counter = Counter(
    name
    for names in merged["genre_names"]
    for name in names
)

top_actors = [name for name, _ in actor_counter.most_common(10)]
top_genres = [name for name, _ in genre_counter.most_common(10)]

print("Top 10 actors:", top_actors)
print("Top 10 genres:", top_genres)

# ---------------------------------------------------------------------
# 5. Add indicator (0/1) columns for those actors and genres
# ---------------------------------------------------------------------

# Create safe column names (no spaces, hyphens -> underscores)
actor_cols = [f"actor_{a.replace(' ', '_')}" for a in top_actors]
genre_cols = [f"genre_{g.replace(' ', '_').replace('-', '_')}" for g in top_genres]

# Actors
for actor, col_name in zip(top_actors, actor_cols):
    merged[col_name] = merged["cast_names"].apply(
        lambda names, actor=actor: int(actor in names)
    )

# Genres
for genre, col_name in zip(top_genres, genre_cols):
    merged[col_name] = merged["genre_names"].apply(
        lambda names, genre=genre: int(genre in names)
    )

# ---------------------------------------------------------------------
# 6. Add release year column
# ---------------------------------------------------------------------

merged["release_year"] = pd.to_datetime(
    merged["release_date"], errors="coerce"
).dt.year

# ---------------------------------------------------------------------
# 7. Build final dataframe and save as CSV
# ---------------------------------------------------------------------

base_cols = ["title", "overview", "release_year"]
final_cols = base_cols + actor_cols + genre_cols

final_df = merged[final_cols].copy()

output_path = "top500_movies_features.csv"
final_df.to_csv(output_path, index=False)

print(f"Saved {len(final_df)} rows with {len(final_df.columns)} columns to {output_path}")
print(final_df.head())
