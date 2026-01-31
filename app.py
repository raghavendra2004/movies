import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. WEB PAGE CONFIG & STYLE ---
st.set_page_config(page_title="MovieTinder AI", page_icon="🎬", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: 3.5em;
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #ff1e2e; color: white; }
    .movie-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING & ROBUST CLEANING ---
@st.cache_data
def load_and_clean_data():
    # Load file
    df = pd.read_csv('movies_metadata.csv', low_memory=False)
    
    # Fix corrupted IDs (this dataset has 3 rows where ID is a date)
    df = df[df['id'].str.isnumeric() == True]
    df['id'] = df['id'].astype(int)
    
    # Select columns
    df = df[['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count', 'poster_path']]
    
    # Drop rows with missing crucial info
    df.dropna(subset=['title', 'overview', 'poster_path'], inplace=True)
    
    # Filter for popularity (Top 3000 movies) to save memory on the server
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    df = df.sort_values('vote_count', ascending=False).head(3000).reset_index(drop=True)
    
    # Convert genres from JSON string to plain text
    def convert_genres(obj):
        try:
            return " ".join([i['name'] for i in ast.literal_eval(obj)])
        except: return ""
    
    df['genres_list'] = df['genres'].apply(convert_genres)
    df['tags'] = df['overview'] + " " + df['genres_list']
    return df

try:
    df = load_and_clean_data()
except FileNotFoundError:
    st.error("Error: 'movies_metadata.csv' not found. Please check your GitHub file names.")
    st.stop()

# --- 3. AI RECOMMENDATION ENGINE ---
@st.cache_resource
def build_similarity_matrix(_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(_df['tags']).toarray()
    return cosine_similarity(vector)

similarity = build_similarity_matrix(df)

def get_recs(movie_title):
    try:
        idx = df[df['title'] == movie_title].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        return [df.iloc[i[0]].title for i in distances[1:6]]
    except: return []

# --- 4. SESSION STATE (The "App Memory") ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'likes' not in st.session_state: st.session_state.likes = []

# --- 5. THE USER INTERFACE ---
st.title("🍿 MovieTinder AI")
st.write("Swipe through movies to get personalized recommendations!")

# Prevent index out of bounds
if st.session_state.idx < len(df):
    movie = df.iloc[st.session_state.idx]
    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"

    # Movie Card Display
    st.markdown(f"""
        <div class="movie-card">
            <img src="{poster_url}" width="250" style="border-radius:15px;">
            <h2>{movie['title']}</h2>
            <p style="color: #ffd700;">⭐ {movie['vote_average']}/10</p>
            <p style="font-style: italic; color: #ccc;">{movie['genres_list']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    with st.expander("Read Plot Summary"):
        st.write(movie['overview'])

    # Tinder Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👎 DISLIKE"):
            st.session_state.idx += 1
            st.rerun()
    with col2:
        if st.button("❤️ LIKE"):
            st.session_state.likes.append(movie['title'])
            st.session_state.idx += 1
            st.rerun()
    with col3:
        if st.button("⏭️ SKIP"):
            st.session_state.idx += 1
            st.rerun()
else:
    st.success("You've seen all the top movies!")

# --- 6. SIDEBAR: LIKES & RECS ---
st.sidebar.header("🎬 My Liked Movies")
for l in st.session_state.likes:
    st.sidebar.write(f"✅ {l}")

if st.session_state.likes:
    st.sidebar.markdown("---")
    st.sidebar.header("✨ Recommended for You")
    last_liked = st.session_state.likes[-1]
    recommendations = get_recs(last_liked)
    for r in recommendations:
        st.sidebar.write(f"⭐ {r}")

if st.sidebar.button("Clear History"):
    st.session_state.likes = []
    st.session_state.idx = 0
    st.rerun()
