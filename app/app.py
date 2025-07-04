import sys
import os
import streamlit as st
import pandas as pd

# Setup for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recommender.recommender import recommend, df

# Constants
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# --- Helpers --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_poster_url(title: str) -> str | None:
    row = df[df["title"] == title]
    if row.empty:
        return None
    path = row.iloc[0].get("poster_path")
    if pd.isna(path):
        return None
    return f"{TMDB_IMG_BASE}{path}"

def get_release_year(release_date_str):
    """Convert a dd-mm-yyyy string to a 4‑digit year."""
    try:
        date = pd.to_datetime(release_date_str, dayfirst=True, errors='coerce')
        return str(date.year) if pd.notnull(date) else "N/A"
    except:
        return "N/A"

def get_movie_info(title):
    movie_data = df[df["title"] == title]
    if movie_data.empty:
        return "No overview", "N/A", None

    overview      = movie_data["overview"].values[0]
    release_year  = movie_data["release_year"].values[0]   # always present
    poster_url    = get_poster_url(title)
    return overview, release_year, poster_url



# --- UI Layout ------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Discover movies similar to your favorites</p>", unsafe_allow_html=True)

# Movie selection
movie_list = df["title"].sort_values().tolist()
selected_movie = st.selectbox("🎞️ Select a movie", movie_list, index=movie_list.index("Inception") if "Inception" in movie_list else 0)

# --- Selected Movie Display -----------------------------------------------
st.markdown("### 🎯 Selected Movie")
overview, release_year, poster_url = get_movie_info(selected_movie)
sel_col1, sel_col2 = st.columns([1, 2])
with sel_col1:
    if poster_url:
        st.image(poster_url, caption=f"{selected_movie} ({release_year})", use_column_width=True)
with sel_col2:
    st.markdown(f"#### {selected_movie} ({release_year})")
    st.write(overview)

# --- Recommendations ------------------------------------------------------
st.markdown("---")
st.markdown("### 🎥 Recommended Movies")

rec_titles = recommend(selected_movie, top_n=5)
rec_cols = st.columns(5)

for col, rec_title in zip(rec_cols, rec_titles):
    rec_overview, rec_year, rec_poster = get_movie_info(rec_title)
    with col:
        with st.container():
            if rec_poster:
                st.image(rec_poster, use_column_width=True)
            st.markdown(f"**{rec_title} ({rec_year})**", unsafe_allow_html=True)
            if rec_overview:
                st.caption(rec_overview[:120] + "...")
