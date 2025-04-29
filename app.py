import streamlit as st
import pandas as pd
import numpy as np
import difflib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('movies.csv')  # Change to your path if needed
    required_columns = ['genres', 'keywords', 'overview', 'tagline', 'cast', 'director', 'title', 'index', 'vote_average', 'runtime', 'release_date']
    for col in required_columns:
        if col not in data.columns:
            raise Exception(f"Column '{col}' not found in dataset. Please check it.")
    
    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    data['release_year'] = data['release_date'].dt.year.fillna(0).astype(int)

    selected_features = ['genres','keywords','overview','tagline','cast','director']
    for feature in selected_features:
        data[feature] = data[feature].fillna("")

    combined = data['genres'] + data['keywords'] + data['overview'] + data['tagline'] + data['cast'] + data['director']
    tfidf = TfidfVectorizer()
    feature_vector = tfidf.fit_transform(combined)
    similarity = cosine_similarity(feature_vector)
    
    return data, similarity

data, similarity = load_data()

mood_genre_map = {
    'happy': ['Comedy', 'Adventure', 'Family'],
    'sad': ['Drama', 'Romance'],
    'excited': ['Action', 'Thriller', 'Science Fiction'],
    'romantic': ['Romance', 'Drama'],
    'scared': ['Horror', 'Thriller'],
    'angry': ['Action', 'War'],
    'inspired': ['Biography', 'History'],
    'bored': ['Mystery', 'Fantasy', 'Animation']
}

mood_emoji_map = {
    'happy': '😊', 'sad': '😢', 'excited': '🤩', 'romantic': '💖',
    'scared': '😱', 'angry': '😠', 'inspired': '✨', 'bored': '🌀', 'default': '🎬'
}

# UI
st.title("🎥 Mood-Based Movie Recommender")

movie_name = st.text_input("Enter your favorite movie name:")
user_mood = st.selectbox("How do you feel today?", list(mood_genre_map.keys()))
year_filter = st.text_input("Want movies after a certain year? (e.g., 2010)", "")

if st.button("Recommend"):
    if user_mood not in mood_genre_map:
        user_mood = 'happy'

    year_filter = int(year_filter) if year_filter.strip().isdigit() else 0

    list_titles = data['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, list_titles)

    if close_matches:
        close_match = close_matches[0]
        movie_index = data[data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[movie_index]))
        sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.subheader(f"Top 10 Recommendations based on *{close_match}* and your mood {mood_emoji_map[user_mood]}")
        count = 0
        for movie in sorted_movies:
            index = movie[0]
            movie_data = data[data.index == index].iloc[0]
            if not any(genre.strip() in movie_data['genres'] for genre in mood_genre_map[user_mood]):
                continue
            if movie_data['release_year'] < year_filter:
                continue

            runtime = movie_data['runtime']
            runtime_cat = "Short" if runtime < 90 else "Medium" if runtime <= 150 else "Long"

            st.markdown(f"**{movie_data['title']}** {mood_emoji_map[user_mood]}")
            st.markdown(f"⭐ {movie_data['vote_average']}/10 | 📅 {movie_data['release_year']} | 🕒 {runtime_cat} ({runtime} mins)")
            st.markdown(f"🎭 Genres: {movie_data['genres']}")
            st.markdown(f"📢 *{movie_data['tagline']}*")
            st.markdown(f"🎬 Director: {movie_data['director']} | Lead Actor: {movie_data['cast'].split(',')[0]}")
            st.markdown("---")

            count += 1
            if count >= 10:
                break
    else:
        st.warning("Movie not found. Showing a random recommendation based on mood.")
        genre_filter = mood_genre_map[user_mood]
        fallback = data[data['genres'].apply(lambda x: any(g in x for g in genre_filter))].sample(1).iloc[0]
        st.markdown(f"**{fallback['title']}** {mood_emoji_map[user_mood]}")
        st.markdown(f"⭐ {fallback['vote_average']} | 📅 {fallback['release_year']}")
        st.markdown(f"🎭 Genres: {fallback['genres']}")
        st.markdown(f"📢 {fallback['tagline']}")
        st.markdown(f"🎬 Director: {fallback['director']} | Cast: {fallback['cast'].split(',')[0]}")
