# 🎬 Movie Recommendation System in Python

An interactive, end-to-end movie recommendation engine built with the **MovieLens 25M dataset** — real data, real techniques, production-ready notebook.

> Type a movie name → see live search results → get 10 hand-picked recommendations instantly.

---

## 📌 What This Project Does

| Feature | Detail |
|---------|--------|
| **Search engine** | TF-IDF + cosine similarity with unigrams & bigrams |
| **Recommendation engine** | Collaborative filtering on 25 million ratings |
| **Interactive UI** | Live `ipywidgets` inside Jupyter — no page reload |
| **Dataset** | [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) — 62K movies, 162K users |

---

## 🗂️ Project Structure

```
Movie Recommendation System in Python/
├── movie_recommendations.ipynb   ← main notebook (run this)
├── ml-25m/
│   ├── movies.csv                ← 62,423 movies with titles & genres
│   ├── ratings.csv               ← 25,000,095 user ratings
│   ├── tags.csv
│   ├── links.csv
│   ├── genome-scores.csv
│   └── genome-tags.csv
└── README.md
```

---

## ⚙️ How It Works

### Step 1 — Data Cleaning (Regex)
Movie titles contain parentheses around years — e.g., `Toy Story (1995)`. We strip everything that isn't a letter, digit, or space using a simple regex so the search engine doesn't get confused by punctuation.

```python
import re

def clean_title(title):
    return re.sub(r"[^a-zA-Z0-9 ]", "", title)

# "Toy Story (1995)" → "Toy Story 1995"
```

### Step 2 — TF-IDF Matrix
We convert all 62,423 cleaned titles into numeric vectors using **TF-IDF** (Term Frequency – Inverse Document Frequency). Setting `ngram_range=(1,2)` captures both single words and pairs (`toy story`, `story 1995`), making searches far more accurate.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])
```

### Step 3 — Cosine Similarity Search
When you type a title, we vectorise your query and find the top-5 closest movie vectors using cosine similarity.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    query_vec = vectorizer.transform([clean_title(title)])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    return movies.iloc[indices].iloc[::-1]
```

### Step 4 — Collaborative Filtering
Once we know *which* movie you searched for, we:
1. Find every user who rated that movie > 4 ⭐ (**similar users**)
2. Find other movies those users also rated > 4 ⭐
3. Keep only movies liked by > 10% of similar users
4. Compare with the *whole* user base to compute a **score**:

$$\text{score} = \frac{\text{% of similar users who liked the movie}}{\text{% of all users who liked the movie}}$$

A high score means a movie is specifically loved by people with your taste — not just generally popular.

```python
def find_similar_movies(movie_id):
    similar_users = ratings[
        (ratings["movieId"] == movie_id) & (ratings["rating"] > 4)
    ]["userId"].unique()

    similar_user_recs = ratings[
        (ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)
    ]["movieId"].value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]

    all_users = ratings[
        (ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)
    ]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    return (
        rec_percentages.sort_values("score", ascending=False)
        .head(10)
        .merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
    )
```

### Step 5 — Interactive Widgets
Both the search box and recommendation output are powered by `ipywidgets` — they update live as you type, no button press required.

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/vidhyawalke/Movie-Recommendation-System-in-Python.git
cd "Movie Recommendation System in Python"
```

### 2. Download the dataset
Download the [MovieLens 25M zip](https://files.grouplens.org/datasets/movielens/ml-25m.zip), unzip it, and place the `ml-25m/` folder inside the project directory.

### 3. Install dependencies
```bash
pip install pandas scikit-learn ipywidgets notebook
```

### 4. Launch the notebook
```bash
jupyter notebook movie_recommendations.ipynb
```

Run all cells from top to bottom. Use the widget in **Cell 5** to search for movies and **Cell 9** to get recommendations.

---

## 📊 Sample Output

Searching for **The Avengers (2012)** returns:

| Score | Title | Genres |
|-------|-------|--------|
| 24.7 | Avengers, The (2012) | Action, Adventure, Sci-Fi |
| 19.6 | Thor: The Dark World (2013) | Action, Adventure, Fantasy |
| 19.5 | Avengers: Age of Ultron (2015) | Action, Adventure, Sci-Fi |
| 17.9 | Iron Man 3 (2013) | Action, Sci-Fi, Thriller |
| 17.8 | Captain America: The First Avenger (2011) | Action, Adventure, Sci-Fi |
| 17.3 | Thor (2011) | Action, Adventure, Fantasy |
| 17.2 | Captain America: The Winter Soldier (2014) | Action, Adventure, Sci-Fi |
| 16.6 | Captain America: Civil War (2016) | Action, Sci-Fi, Thriller |
| 15.9 | Ant-Man (2015) | Action, Adventure, Sci-Fi |
| 15.7 | Iron Man 2 (2010) | Action, Adventure, Sci-Fi |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas** — data loading and manipulation
- **scikit-learn** — TF-IDF vectorisation and cosine similarity
- **NumPy** — fast array operations
- **ipywidgets** — live interactive UI inside Jupyter
- **Jupyter Notebook** — interactive computing environment

---

## 📈 Ideas for Extension

- Add a **genre filter** dropdown widget
- Use `genome-scores.csv` for tag-relevance weighting
- Build a **Streamlit** or **Gradio** web app for browser-based access
- Try **matrix factorisation** (SVD) for a model-based approach

---

## 📄 Dataset Credit

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4):19:1–19:19.  
[https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)

---

## 👤 Author

**Vidhya Walke**  
GitHub: [@vidhyawalke](https://github.com/vidhyawalke)
