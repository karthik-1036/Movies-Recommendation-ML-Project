import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load datasets
movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
credits_df = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge on title
df = movies_df.merge(credits_df, on='title')

# Select useful columns
# keep release_date too
df = df[['movie_id', 'title', 'overview',
         'genres', 'keywords', 'cast', 'crew',
         'release_date']]           # 👈 add this

# add a clean year column (string)
df['release_year'] = pd.to_datetime(
    df['release_date'], dayfirst=True, errors='coerce'
).dt.year

# Convert year to clean string format (no decimals)
df['release_year'] = df['release_year'].apply(
    lambda x: str(int(x)) if pd.notnull(x) else "N/A"
)


# Parse stringified JSON
def parse(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

df['genres'] = df['genres'].apply(parse)
df['keywords'] = df['keywords'].apply(parse)

def get_top_cast(cast_str):
    try:
        return [i['name'] for i in ast.literal_eval(cast_str)[:3]]
    except:
        return []

df['cast'] = df['cast'].apply(get_top_cast)

def get_director(crew_str):
    try:
        for i in ast.literal_eval(crew_str):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except:
        return []

df['crew'] = df['crew'].apply(get_director)

# Combine all features into a single string
df['tags'] = df['overview'].fillna('') + ' ' + df['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
             df['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
             df['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
             df['crew'].apply(lambda x: ' '.join(x))

# Lowercase + remove stopwords
stop_words = set(stopwords.words('english'))
df['tags'] = df['tags'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_words]))

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
vectors = tfidf.fit_transform(df['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Mapping title → dataframe index
movie_indices = pd.Series(df.index, index=df['title'])

# Quick helpers for evaluation:
movie_titles  = df['title'].tolist()
movie_vectors = {title: vectors[idx]            # keep it sparse (efficient)
                 for idx, title in enumerate(movie_titles)}

# Recommendation function
def recommend(movie_title, top_n=5):
    if movie_title not in df['title'].values:
        return f"❌ '{movie_title}' not found."
    
    idx = df[df['title'] == movie_title].index[0]
    distances = list(enumerate(similarity[idx]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    return [df.iloc[i[0]]['title'] for i in movies]

__all__ = [
    "recommend",
    "df",
    "similarity",      # ← add these two lines
    "movie_indices"
]
