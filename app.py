import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- WEB PAGE STYLING (CSS) ---
st.set_page_config(page_title="MovieTinder", page_icon="🎬", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #121212; color: white; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #E50914; /* Netflix Red */
        color: white;
        border: none;
    }
    .stButton>button:hover { background-color: #ff0a16; color: white; }
    .movie-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        text-align: center;
    }
    .poster-img { border-radius: 10px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('movies_metadata.csv', low_memory=False)
    # Keep only important rows and filter for speed
    df = df[['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count', 'poster_path']]
    df.dropna(subset=['title', 'overview', 'poster_path'], inplace=True)
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    df = df.sort_values('vote_count', ascending=False).head(3000).reset_index(drop=True)
    return df

df = load_data()

# --- HELPER: GET POSTER URL ---
def get_poster_url(path):
    return f"https://image.tmdb.org/t/p/w500{path}"

# --- UI LOGIC ---
if 'idx' not in st.session_state: st.session_state.idx = 0
if 'likes' not in st.session_state: st.session_state.likes = []

movie = df.iloc[st.session_state.idx]

# DISPLAY MOVIE
st.markdown(f"""
    <div class="movie-card">
        <img src="{get_poster_url(movie['poster_path'])}" width="300" class="poster-img">
        <h1 style='color: white;'>{movie['title']}</h1>
        <p style='color: #bbb;'>⭐ {movie['vote_average']} | {movie['genres'][:30]}...</p>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer
st.info(movie['overview'])

# BUTTONS
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("👎 Pass"): 
        st.session_state.idx += 1
        st.rerun()
with c2:
    if st.button("❤️ Like"):
        st.session_state.likes.append(movie['title'])
        st.session_state.idx += 1
        st.rerun()
with c3:
    if st.button("⏭️ Skip"):
        st.session_state.idx += 1
        st.rerun()

# SIDEBAR (YOUR SAVED LIST)
st.sidebar.title("🍿 My List")
for liked in st.session_state.likes:
    st.sidebar.write(f"• {liked}")