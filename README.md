# 🎬 Movie Recommendation System

This is a content-based movie recommender system built using NLP and machine learning. It suggests movies based on their **plot**, **genre**, **cast**, and **keywords**, using **TF-IDF vectorization** and **cosine similarity** to find similar titles. The frontend is powered by **Streamlit**, with movie posters and descriptions fetched from TMDb.


## 🧠 How It Works

### 🔍 Recommendation Logic

The system recommends movies using:
- **Textual data** (overview, genres, cast, keywords)
- **TF-IDF vectorization** to create feature vectors
- **Cosine similarity** to find similar movies

### 📦 Dataset Used

We use the **TMDB 5000 Movies Dataset**, which contains:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

> Source: [Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 🖥️ UI Features (Built with Streamlit)

- 🔍 Select a movie from dropdown
- 🖼️ View poster, title, and overview of the selected movie
- 🎥 See **top 5 similar movies** with posters, short description, and release year

## 📁 Project Structure

```
movie-recommender/
│
├── app/
│   └── app.py                  # Streamlit UI
│
├── recommender/
│   └── recommender.py          # Core recommendation logic
│
├── data/
│   ├── tmdb_5000_movies.csv    # Movie metadata
│   └── tmdb_5000_credits.csv   # Cast & crew data
│
├── evaluate.py                 # Evaluation script for similarity metrics
├── requirements.txt            # Dependencies
├── download_nltk_resources.py # Downloads NLTK stopwords
└── README.md                   # You're here
```

## 🧪 Evaluation

You can evaluate recommendation quality using cosine similarity.

To run:

```bash
python evaluate.py
```

Sample output:

```
🎯 Inception – avg cosine similarity of top 5: 0.1489
  Hesher                         0.1641
  Don Jon                        0.1618
  ...
```

## ⚙️ Installation

### ✅ Prerequisites

- Python 3.10+
- Anaconda (recommended)

### 📦 Create Conda Environment

```bash
conda create -n movie-recommender python=3.10 -y
conda activate movie-recommender
```

### 📥 Install Requirements

```bash
pip install -r requirements.txt
python download_nltk_resources.py
```

## ▶️ Running the App

```bash
streamlit run app/app.py
```

Open browser at: http://localhost:8501

## 📊 Future Improvements

- 🔄 Add collaborative filtering model
- 🔮 Use BERT-based embeddings (e.g., Sentence Transformers)
- 🎯 Incorporate user ratings
- 🌐 Deploy to Streamlit Cloud / Hugging Face Spaces

## 🛠️ Tech Stack

- 🐍 Python
- 📊 Pandas, Scikit-learn, NLTK
- 🌐 Streamlit
- 🧠 TF-IDF Vectorizer, Cosine Similarity
- 🖼️ TMDb API (for poster fetching)

## 📌 Credits

- Dataset: [Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Poster Fetching: [TMDb](https://www.themoviedb.org/)

## 📜 License

This project is licensed under the MIT License.
