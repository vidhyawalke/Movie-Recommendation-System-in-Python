<p align="center">
  <img src="banner.svg" alt="Movie Recommendation System in Python" width="100%"/>
</p>

<h1 align="center">🎬 Movie Recommendation System in Python</h1>

<p align="center">
  An interactive, end-to-end movie recommendation engine built with the MovieLens 25M dataset.<br/>
  Real data. Real techniques. Production-ready notebook.
</p>


---

## Overview

This project implements a two-part interactive movie recommendation system entirely in Python. The first part is a **content-based search engine** that uses TF-IDF vectorisation and cosine similarity to match a typed movie title against a corpus of 62,423 entries in real time. The second part is a **collaborative filtering recommendation engine** that mines 25 million user ratings to surface films that are disproportionately popular among audiences who share the same taste as you.

Both components are wired to live Jupyter widgets so the entire experience, from search to recommendation, updates as you type with no manual cell re-execution required.

---

## Acknowledgement

This project was guided by the structured walkthrough provided by **[Dataquest](https://www.dataquest.io/)** as part of their project-based learning curriculum:

> [Build a Movie Recommendation System in Python](https://app.dataquest.io/c/93/m/99994/build-a-movie-recommendation-system-in-python/1/introduction) — Dataquest Intermediate Python Project

The dataset used throughout this project is the **MovieLens 25M** dataset, published and maintained by the GroupLens research group at the University of Minnesota:

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems, 5(4): 19:1–19:19. [https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)

---

## Dataset

| File | Description | Size |
|------|-------------|------|
| `movies.csv` | 62,423 movies with title and genre metadata | 3 MB |
| `ratings.csv` | 25,000,095 ratings from 162,541 users on a 0.5 to 5.0 scale | 650 MB |
| `tags.csv` | User-generated tags applied to movies | 39 MB |
| `genome-scores.csv` | Tag relevance scores for 13,816 movies | 435 MB |
| `links.csv` | IMDb and TMDb identifiers for each movie | 1.4 MB |

Download the full dataset as a single archive from GroupLens:  
[https://files.grouplens.org/datasets/movielens/ml-25m.zip](https://files.grouplens.org/datasets/movielens/ml-25m.zip)

Place the extracted `ml-25m/` folder in the project root before running the notebook.

---

## Project Structure

```
Movie Recommendation System in Python/
├── movie_recommendations.ipynb
├── banner.svg
├── .gitignore
└── README.md
```

The `ml-25m/` directory is registered in `.gitignore` and is not tracked in git due to file size limits.

---

## Methodology

### Part 1: Content-Based Movie Search

**Data Cleaning with Regular Expressions**

Movie titles in the dataset include release year annotations in parentheses, such as `Toy Story (1995)`. Before vectorisation, all non-alphanumeric characters (excluding spaces) are removed using regex substitution. This prevents punctuation from fragmenting tokens while preserving the release year as an independent token.

```python
import re

def clean_title(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 ]", "", title)

# Result: "Toy Story (1995)" becomes "Toy Story 1995"
```

**TF-IDF Vectorisation**

Cleaned titles are converted into sparse numerical vectors using `TfidfVectorizer`. The parameter `ngram_range=(1, 2)` instructs the vectoriser to generate both unigrams and bigrams. The title *Toy Story 1995* generates unigrams (`toy`, `story`, `1995`) as well as bigrams (`toy story`, `story 1995`), substantially improving differentiation between films with similar wording.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

Here $t$ represents a term, $d$ denotes a document (title), $N$ represents the total corpus size, and $\text{DF}(t)$ is the document frequency of term $t$.

**Cosine Similarity Search**

During search execution, the query string is cleaned and mapped into the fitted TF-IDF feature space. Cosine similarity is computed between the query vector and all matrix rows:

$$\cos(\theta) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

The top 5 titles ranked by similarity score are extracted efficiently in $O(n)$ time using `numpy.argpartition`.

### Part 2: Collaborative Filtering Recommendation Engine

**Step 1: Identifying Similar Users**

For a chosen movie, the system queries all users who assigned it a rating strictly greater than 4.0. This isolates a cohort of users with demonstrated affinity for the title.

**Step 2: Candidate Recommendation Pool**

Among this cohort, all other films rated higher than 4.0 are aggregated. The frequency of each film is divided by the total count of similar users to determine the cohort endorsement percentage. Films endorsed by fewer than 10% of the cohort are filtered out to eliminate low-frequency noise.

**Step 3: Relative Score Computation**

The endorsement percentage is also calculated across the global user population for each candidate film. Dividing the cohort percentage by the global percentage yields the recommendation score:

$$\text{score}(m) = \frac{P(\text{similar users liked } m)}{P(\text{all users liked } m)}$$

A high score indicates that a film is disproportionately favoured by users who loved the target film, filtering out broadly popular blockbusters that lack topical specificity.

---

## Sample Output

Recommendations produced when searching for **The Avengers (2012)**:

| Rank | Recommendation Score | Title | Genres |
|:---:|:---:|:---|:---|
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

The system anchors on *The Avengers* at rank 1 with score 24.72 and populates the remaining slots exclusively with interconnected Marvel Cinematic Universe films, validating the precision of the relative scoring metric.

---

## Quickstart

**1. Clone Repository**

```bash
git clone https://github.com/vidhyawalke/Movie-Recommendation-System-in-Python.git
cd "Movie Recommendation System in Python"
```

**2. Download Dataset**

```bash
# macOS or Linux
curl -O https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip

# Windows PowerShell
Invoke-WebRequest -Uri https://files.grouplens.org/datasets/movielens/ml-25m.zip -OutFile ml-25m.zip
Expand-Archive ml-25m.zip
```

**3. Install Dependencies**

```bash
pip install pandas scikit-learn numpy ipywidgets notebook
```

**4. Run Jupyter Notebook**

```bash
jupyter notebook movie_recommendations.ipynb
```

Execute cells sequentially. The interactive search widget is initialized in Cell 6, and the complete recommendation widget is in Cell 13.

---

## Tech Stack

| Component | Technology | Role |
|:---|:---|:---|
| Corpus Processing | pandas | Data cleaning and tabular aggregation |
| Vectorisation | scikit-learn | TfidfVectorizer with unigrams and bigrams |
| Metric Computation | scikit-learn | Cosine similarity matrix evaluation |
| Array Indexing | NumPy | Fast top-k partition via argpartition |
| Interactive Interface | ipywidgets | Real-time text listening and reactive output |
| Runtime Environment | Jupyter Notebook | Interactive execution and analysis |

---

## Future Enhancements

1. **Genre Filtering Controls**: Introduce a multi-select genre widget to narrow down collaborative filtering outputs by specific categories.
2. **Tag Genome Integration**: Leverage `genome-scores.csv` to combine collaborative filtering signals with 1,128 semantic tag vectors.
3. **Matrix Factorisation**: Implement Truncated SVD or Alternating Least Squares (ALS) to model latent user-item interaction spaces.
4. **Standalone Web Application**: Package the inference pipeline into a lightweight Streamlit or FastAPI service for direct web deployment.

---

## Getting Help

If you encounter issues or have questions, please open an issue in the [GitHub Issue Tracker](https://github.com/vidhyawalke/Movie-Recommendation-System-in-Python/issues).

---

## Maintainer

Built and maintained by [Vidhya Walke](https://github.com/vidhyawalke).  
Contributions and feedback are welcome through pull requests and issues.
