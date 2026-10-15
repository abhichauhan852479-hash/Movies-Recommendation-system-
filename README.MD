# 🎬 Movie Recommendation System

A Python-based movie recommendation web application that suggests similar movies using content-based filtering and cosine similarity.

## Features
- Select a movie from a dropdown menu
- Get five similar movie recommendations
- Compare movie genres, keywords, cast, directors, and descriptions
- View movie descriptions and content similarity scores

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit

## Machine Learning Method
The application uses CountVectorizer to convert movie information into numerical features and cosine similarity to identify similar movies.

## Dataset
This project uses the TMDB 5000 Movie Dataset.

Download it from:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Place the two files below inside a folder named `data`:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## How to Run

1. Install Python 3.
2. Download the dataset and place both CSV files in the `data` folder.
3. Install the required packages:

   `python -m pip install -r requirements.txt`

4. Run the application:

   `python -m streamlit run app.py`

5. Open the local URL shown in the terminal.

## Project Purpose
This project demonstrates a beginner-friendly content-based movie recommendation system built with Python and machine learning.