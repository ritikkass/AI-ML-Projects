## 2. Movie Recommendation System

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('movies.csv')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movie(title):
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]]['title'] for i in scores]

print(recommend_movie('Inception'))
