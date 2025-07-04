from recommender.recommender import recommend, movie_vectors, movie_titles
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from recommender.recommender import recommend, similarity, movie_indices

def evaluate_recommendations(title, top_n=5):
    if title not in movie_indices:
        print(f"'{title}' not in data")
        return
    idx = movie_indices[title]
    rec_titles = recommend(title, top_n)
    sims = [similarity[idx, movie_indices[t]] for t in rec_titles]
    print(f"\n🎯 {title} – avg cosine similarity of top {top_n}: {sum(sims)/len(sims):.4f}")
    for r, s in zip(rec_titles, sims):
        print(f"  {r:40s}  {s:.4f}")

if __name__ == "__main__":
    evaluate_recommendations("The Avengers", 5)

