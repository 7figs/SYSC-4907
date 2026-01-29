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

# Select top 500 movies
movies_top = movies.sort_values(by="vote_average", ascending=False).head(500).copy()

# Merge movies and credits
merged = movies_top.merge(credits, left_on="id", right_on="movie_id", how="left")

if "title_x" in merged.columns:
    merged = merged.rename(columns={"title_x": "title"})

def parse_list_column(s):
    if pd.isna(s) or s == "": return []
    try: return ast.literal_eval(s)
    except: return []

# --- EXTRACTING METADATA ---
merged["genre_names"] = merged["genres"].apply(lambda x: [d.get("name") for d in parse_list_column(x)])
merged["cast_names"] = merged["cast"].apply(lambda x: [d.get("name") for d in parse_list_column(x)])
merged["studio_names"] = merged["production_companies"].apply(lambda x: [d.get("name") for d in parse_list_column(x)])
merged["keyword_list"] = merged["keywords"].apply(lambda x: [d.get("name") for d in parse_list_column(x)])
merged["country_names"] = merged["production_countries"].apply(lambda x: [d.get("name") for d in parse_list_column(x)])

def get_director(crew_str):
    crew = parse_list_column(crew_str)
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name")
    return "Unknown"

merged["director_name"] = merged["crew"].apply(get_director)

# --- IDENTIFYING TOP FEATURES ---
# We use the top frequent items to avoid creating thousands of columns
top_actors = [n for n, _ in Counter(n for l in merged["cast_names"] for n in l).most_common(15)]
top_studios = [n for n, _ in Counter(n for l in merged["studio_names"] for n in l).most_common(15)]
top_directors = [n for n, _ in Counter(merged["director_name"]).most_common(12)]
top_keywords = [n for n, _ in Counter(n for l in merged["keyword_list"] for n in l).most_common(20)]
top_countries = [n for n, _ in Counter(n for l in merged["country_names"] for n in l).most_common(5)]
all_genres = sorted({n for l in merged["genre_names"] for n in l})

# --- FEATURE ENGINEERING (NO BUDGET/REVENUE) ---

indicator_cols = []

def add_indicators(df, source_col, items, prefix):
    for item in items:
        col_name = f"{prefix}_{item.replace(' ', '_').replace('-', '_')}"
        df[col_name] = df[source_col].apply(lambda x: int(item in x) if isinstance(x, list) else int(item == x))
        indicator_cols.append(col_name)

add_indicators(merged, "cast_names", top_actors, "actor")
add_indicators(merged, "genre_names", all_genres, "genre")
add_indicators(merged, "studio_names", top_studios, "studio")
add_indicators(merged, "director_name", top_directors, "dir")
add_indicators(merged, "keyword_list", top_keywords, "kw")
add_indicators(merged, "country_names", top_countries, "country")

# Language & Meta
merged["is_english"] = (merged["original_language"] == "en").astype(int)
merged["release_year"] = pd.to_datetime(merged["release_date"], errors="coerce").dt.year.fillna(0)

# --- FINAL SELECTION ---
numeric_cols = ["popularity", "vote_count", "runtime", "release_year", "is_english"]
base_info = ["title", "overview"]

final_df = merged[base_info + numeric_cols + indicator_cols].copy()

# Clean up column names
final_df.columns = [c.replace(".", "").replace(" ", "_") for c in final_df.columns]

# Save and SQL Update
output_path = "top500_content_features.csv"
final_df.to_csv(output_path, index=False)

conn = sqlite3.connect(db_path)
final_df.to_sql(const.MOVIES_TABLE, conn, if_exists="replace", index=False)
conn.commit()
conn.close()

print(f"File '{output_path}' created with {len(final_df.columns)} content-based features.")