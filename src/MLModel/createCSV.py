import sqlite3
import pandas as pd
import ast
from collections import Counter
import sys
import os

# Allow importing const from parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import const

movies_path = "tmdb_5000_movies.csv"
credits_path = "tmdb_5000_credits.csv"
db_path = const.DB_PATH

# Load CSVs
movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

# Select top 500 movies by vote_average
movies_top = movies.sort_values(by="vote_average", ascending=False).head(500).copy()

# Merge movies and credits
merged = movies_top.merge(
    credits,
    left_on="id",
    right_on="movie_id",
    how="left"
)

# Clean up duplicate title columns after merge
if "title_x" in merged.columns:
    merged = merged.rename(columns={"title_x": "title"})
if "title_y" in merged.columns:
    merged = merged.drop(columns=["title_y"])

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

# Extract genre names
merged["genre_names"] = merged["genres"].apply(
    lambda x: [
        d.get("name") for d in parse_list_column(x)
        if isinstance(d, dict) and "name" in d
    ]
)

# Extract cast names
merged["cast_names"] = merged["cast"].apply(
    lambda x: [
        d.get("name") for d in parse_list_column(x)
        if isinstance(d, dict) and "name" in d
    ]
)

# Count actors (kept as-is)
actor_counter = Counter(
    name
    for names in merged["cast_names"]
    for name in names
)

top_actors = [name for name, _ in actor_counter.most_common(10)]

# NEW: get ALL unique genres appearing in top 500 movies
all_genres = sorted({
    name
    for names in merged["genre_names"]
    for name in names
})

print("Top 10 actors:", top_actors)
print("All genres in top 500 movies:", all_genres)
print(f"Total unique genres: {len(all_genres)}")

# Create column names
actor_cols = [f"actor_{a.replace(' ', '_')}" for a in top_actors]
genre_cols = [
    f"genre_{g.replace(' ', '_').replace('-', '_')}"
    for g in all_genres
]

# Actor indicator columns
for actor, col_name in zip(top_actors, actor_cols):
    merged[col_name] = merged["cast_names"].apply(
        lambda names, actor=actor: int(actor in names)
    )

# Genre indicator columns
for genre, col_name in zip(all_genres, genre_cols):
    merged[col_name] = merged["genre_names"].apply(
        lambda names, genre=genre: int(genre in names)
    )

# Extract release year
merged["release_year"] = pd.to_datetime(
    merged["release_date"], errors="coerce"
).dt.year

# Final dataframe
base_cols = ["title", "overview", "release_year"]
final_cols = base_cols + actor_cols + genre_cols
final_df = merged[final_cols].copy()

# Save to CSV
output_path = "top500_movies_features.csv"
final_df.to_csv(output_path, index=False)

print(f"Saved {len(final_df)} rows with {len(final_df.columns)} columns to {output_path}")
print(final_df.head())

# Clean column names (remove dots)
df = pd.read_csv(output_path)
df.columns = df.columns.str.replace(".", "", regex=False)
df.to_csv(output_path, index=False)

# Write to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

delete_query = f"DELETE FROM {const.MOVIES_TABLE};"
cursor.execute(delete_query)
conn.commit()
conn.close()

df = pd.read_csv(output_path)
conn = sqlite3.connect(db_path)
df.to_sql(const.MOVIES_TABLE, conn, if_exists="replace", index=False)
conn.commit()
conn.close()
