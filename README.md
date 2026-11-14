<p align="center">
  <img src="banner.png" alt="Movie Recommendation System" width="800"/>
</p>

<h1 align="center">🎬 Movie Recommendation System in Python</h1>

<p align="center">
  An interactive, end-to-end movie recommendation engine built with the MovieLens 25M dataset.<br/>
  Real data. Real techniques. Production-ready notebook.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-TF--IDF%20%7C%20Cosine%20Similarity-orange?style=flat-square&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Dataset-MovieLens%2025M-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat-square&logo=jupyter" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

---

## Overview

This project implements a two-part interactive movie recommendation system entirely in Python. The first part is a **content-based search engine** that uses TF-IDF vectorisation and cosine similarity to match a typed movie title against a corpus of 62,423 entries in real time. The second part is a **collaborative filtering recommendation engine** that mines 25 million user ratings to surface films that are disproportionately popular among audiences who share the same taste as you.

Both components are wired to live Jupyter widgets so the entire experience — from search to recommendation — updates as you type, with no manual cell re-execution required.

---

## Acknowledgement

This project was guided by the structured walkthrough provided by **[Dataquest](https://www.dataquest.io/)** as part of their project-based learning curriculum:

> [Build a Movie Recommendation System in Python](https://app.dataquest.io/c/93/m/99994/build-a-movie-recommendation-system-in-python/1/introduction) — Dataquest Intermediate Python Project (13 steps)

The dataset used throughout this project is the **MovieLens 25M** dataset, published and maintained by the GroupLens research group at the University of Minnesota:

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems, 5(4): 19:1–19:19. [https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)

---

## Dataset

| File | Description | Size |
|------|-------------|------|
| `movies.csv` | 62,423 movies with title and genre metadata | 3 MB |
| `ratings.csv` | 25,000,095 ratings from 162,541 users (scale: 0.5 to 5.0) | 650 MB |
| `tags.csv` | User-generated tags applied to movies | 39 MB |
| `genome-scores.csv` | Tag relevance scores for 13,816 movies | 435 MB |
| `links.csv` | IMDb and TMDb identifiers for each movie | 1.4 MB |

Download the full dataset as a single zip archive:
[https://files.grouplens.org/datasets/movielens/ml-25m.zip](https://files.grouplens.org/datasets/movielens/ml-25m.zip)

Place the extracted `ml-25m/` folder in the project root before running the notebook.

---

## Project Structure

```
Movie Recommendation System in Python/
├── movie_recommendations.ipynb   ← main notebook with all outputs
├── banner.png                    ← project banner
├── .gitignore                    ← excludes large dataset files
└── README.md
```

The `ml-25m/` dataset folder is listed in `.gitignore` and is not committed to the repository. Reviewers must download it separately using the link above.

---

## Methodology

### Part 1: Content-Based Movie Search

**Data Cleaning with Regular Expressions**

Movie titles in the dataset include year annotations in parentheses — for example, `Toy Story (1995)`. Before vectorisation, all non-alphanumeric characters (excluding spaces) are stripped using a regex substitution. This prevents punctuation from fragmenting tokens and ensures that the year is treated as a searchable term.

```python
import re

def clean_title(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 ]", "", title)

# "Toy Story (1995)"  ->  "Toy Story 1995"
```

**TF-IDF Vectorisation**

Cleaned titles are converted into sparse numerical vectors using scikit-learn's `TfidfVectorizer`. The `ngram_range=(1, 2)` parameter instructs the vectoriser to generate both unigrams and bigrams. This means the title *Toy Story 1995* yields individual tokens (`toy`, `story`, `1995`) as well as compound tokens (`toy story`, `story 1995`), substantially improving discrimination between similarly named films.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

where $t$ is a term, $d$ is a document (title), $N$ is the total number of titles, and $\text{DF}(t)$ is the number of titles containing term $t$.

**Cosine Similarity Search**

At query time the user's input is cleaned and transformed into the same vector space. Cosine similarity is computed between the query vector and every title vector in the matrix. The five titles with the highest similarity scores are returned.

$$\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

A value of 1.0 indicates identical direction (perfect match); 0 indicates orthogonality (no shared terms).

### Part 2: Collaborative Filtering Recommendation Engine

**Finding Similar Users**

For a given target film, the system identifies every user who assigned it a rating strictly greater than 4.0 (a threshold that selects genuine enthusiasm rather than neutral approval). These users form the *similar user cohort*.

**Candidate Movie Pool**

Among the similar user cohort, the system tallies every other highly-rated film and divides by cohort size to obtain a *similar user percentage* per movie. Only films cleared by more than 10% of the cohort are retained as candidates, filtering out noise from low-signal ratings.

**Niche Scoring**

The same percentage is computed across the full user population for each candidate film. The two percentages are combined into a *recommendation score*:

$$\text{score}(m) = \frac{P(\text{similar users liked } m)}{P(\text{all users liked } m)}$$

A high score indicates that a film is specifically overrepresented in the taste profile of people who liked the target — not merely a broadly popular title. The top 10 candidates by score are returned as final recommendations.

---

## Sample Output

The following recommendations are produced when searching for **The Avengers (2012)**:

| Rank | Score | Title | Genres |
|------|-------|-------|--------|
| 1 | 24.72 | Avengers, The (2012) | Action, Adventure, Sci-Fi, IMAX |
| 2 | 19.61 | Thor: The Dark World (2013) | Action, Adventure, Fantasy, IMAX |
| 3 | 19.49 | Avengers: Age of Ultron (2015) | Action, Adventure, Sci-Fi |
| 4 | 17.87 | Iron Man 3 (2013) | Action, Sci-Fi, Thriller, IMAX |
| 5 | 17.84 | Captain America: The First Avenger (2011) | Action, Adventure, Sci-Fi |
| 6 | 17.30 | Thor (2011) | Action, Adventure, Drama, Fantasy |
| 7 | 17.18 | Captain America: The Winter Soldier (2014) | Action, Adventure, Sci-Fi, IMAX |
| 8 | 16.65 | Captain America: Civil War (2016) | Action, Sci-Fi, Thriller |
| 9 | 15.87 | Ant-Man (2015) | Action, Adventure, Sci-Fi |
| 10 | 15.65 | Iron Man 2 (2010) | Action, Adventure, Sci-Fi, Thriller |

The score of 24.72 for The Avengers itself confirms the system is anchoring on the correct movie. All remaining recommendations are thematically coherent MCU titles — a strong validation of the collaborative filtering signal.

---

## Quickstart

**1. Clone the repository**

```bash
git clone https://github.com/vidhyawalke/Movie-Recommendation-System-in-Python.git
cd "Movie Recommendation System in Python"
```

**2. Download the dataset**

```bash
# Linux / macOS
curl -O https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip

# Windows (PowerShell)
Invoke-WebRequest -Uri https://files.grouplens.org/datasets/movielens/ml-25m.zip -OutFile ml-25m.zip
Expand-Archive ml-25m.zip
```

**3. Install dependencies**

```bash
pip install pandas scikit-learn numpy ipywidgets notebook
```

**4. Launch the notebook**

```bash
jupyter notebook movie_recommendations.ipynb
```

Run all cells from top to bottom. The interactive search widget is in **Cell 5** and the full recommendation widget is in **Cell 9**.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data manipulation | pandas |
| Vectorisation | scikit-learn TfidfVectorizer |
| Similarity computation | scikit-learn cosine_similarity |
| Numerical operations | NumPy |
| Interactive UI | ipywidgets |
| Notebook environment | Jupyter Notebook |

---

## Potential Extensions

The scoring framework developed here is extensible in several directions:

- **Genre-aware filtering**: Add a dropdown widget that filters recommendations by genre, using the pipe-delimited genre field already present in `movies.csv`.
- **Tag-based content signals**: Incorporate `genome-scores.csv`, which provides continuous relevance scores for 1,128 tags across 13,816 movies, to weight the collaborative signal with semantic similarity.
- **Matrix factorisation**: Replace the percentage-based heuristic with a proper latent factor model (e.g. SVD via `scipy.sparse.linalg`) for more generalised user-to-movie projections.
- **Web deployment**: Wrap the two functions in a Streamlit or Gradio app for a shareable browser-based interface without requiring Jupyter.

---

## Getting Help

If you encounter issues or have questions, please open a [GitHub Issue](https://github.com/vidhyawalke/Movie-Recommendation-System-in-Python/issues).

---

## Maintainer

Built and maintained by [Vidhya Walke](https://github.com/vidhyawalke).  
Contributions, bug reports, and suggestions are welcome via pull request or issue.
