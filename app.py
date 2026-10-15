
import ast

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 1. Page configuration
# -------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -------------------------------
# 2. Load and prepare movie data
# -------------------------------

@st.cache_resource
def load_movie_data():

    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    # Keep the columns needed for recommendations
    credits = credits[["movie_id", "cast", "crew"]]

    # Match credits to the correct movie
    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="inner"
    )

    movies = movies[
        ["id", "title", "genres", "keywords",
         "overview", "cast", "crew"]
    ]

    # Remove movies without titles or descriptions
    movies = movies.dropna(subset=["title", "overview"])
    movies = movies.drop_duplicates(subset=["id"])
    movies = movies.reset_index(drop=True)

    # Convert JSON-like text into lists of names
    def get_names(value):
        try:
            items = ast.literal_eval(value)
            return [
                item["name"].replace(" ", "")
                for item in items
                if "name" in item
            ]
        except (ValueError, SyntaxError, TypeError):
            return []

    # Extract the director's name
    def get_director(value):
        try:
            people = ast.literal_eval(value)
            for person in people:
                if person.get("job") == "Director":
                    return person["name"].replace(" ", "")
        except (ValueError, SyntaxError, TypeError):
            pass

        return ""

    # Prepare the movie features
    movies["genres"] = movies["genres"].apply(get_names)
    movies["keywords"] = movies["keywords"].apply(get_names)
    movies["cast"] = movies["cast"].apply(
        lambda value: get_names(value)[:3]
    )
    movies["crew"] = movies["crew"].apply(get_director)

    # Combine information into one text field
    movies["tags"] = (
        movies["genres"].apply(lambda x: " ".join(x)) + " " +
        movies["keywords"].apply(lambda x: " ".join(x)) + " " +
        movies["cast"].apply(lambda x: " ".join(x)) + " " +
        movies["crew"] + " " +
        movies["overview"].fillna("")
    )

    # Convert text into numerical features
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(movies["tags"])

    # Compare movies using cosine similarity
    similarity = cosine_similarity(vectors)

    return movies, similarity


# -------------------------------
# 3. Recommend similar movies
# -------------------------------

def recommend_movies(movie_title, movies, similarity):

    matches = movies.index[
        movies["title"].str.casefold() == movie_title.casefold()
    ].tolist()

    if not matches:
        return pd.DataFrame()

    movie_index = matches[0]

    scores = list(enumerate(similarity[movie_index]))

    # Sort by similarity, highest first
    scores = sorted(
        scores,
        key=lambda item: item[1],
        reverse=True
    )

    # Skip the selected movie itself
    scores = [
        item for item in scores
        if item[0] != movie_index
    ][:5]

    indices = [item[0] for item in scores]

    result = movies.iloc[indices][
        ["title", "overview", "genres"]
    ].copy()

    result["similarity"] = [
        scores[i][1] for i in range(len(scores))
    ]

    return result


# -------------------------------
# 4. Website interface
# -------------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Find movies similar to the movies you already enjoy."
)

st.caption(
    "Recommendations are based on movie genres, keywords, "
    "cast, director, and plot descriptions."
)

try:
    with st.spinner("Loading movies and preparing recommendations..."):
        movies, similarity = load_movie_data()

    st.success(f"Ready! {len(movies)} movies loaded.")

    selected_movie = st.selectbox(
        "Choose a movie you like",
        movies["title"].tolist()
    )

    if st.button("Recommend Movies", type="primary"):

        recommendations = recommend_movies(
            selected_movie,
            movies,
            similarity
        )

        st.subheader("🍿 Recommended for You")

        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            for _, movie in recommendations.iterrows():

                with st.container(border=True):
                    st.subheader(movie["title"])

                    genres = " • ".join(movie["genres"])
                    st.caption(genres if genres else "Genre unavailable")

                    overview = movie["overview"]

                    st.write(
                        overview[:350] + "..."
                        if len(overview) > 350
                        else overview
                    )

                    st.caption(
                        f"Content similarity: "
                        f"{movie['similarity']:.0%}"
                    )

except FileNotFoundError:
    st.error(
        "Dataset files not found. Check that both CSV files "
        "are inside the data folder."
    )
except Exception as error:
    st.error(f"Something went wrong: {error}")
    st.info(
        "Check your CSV files and make sure the required "
        "libraries are installed."
    )
