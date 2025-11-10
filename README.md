# 🎥 Movie Recommendation System using Cosine Similarity

## 🧠 Overview

The **Movie Recommendation System** suggests movies to users based on their viewing history or preferences using **Cosine Similarity**.
It analyzes movie features such as genres, keywords, cast, and crew to calculate similarity scores and recommend the most relevant titles.

---

## 🎯 Objective

To build a **content-based recommendation system** that:

* Analyzes similarities between movies.
* Recommends top related movies based on a selected title.
* Helps users discover new content similar to their interests.

---

## 🧩 Technologies Used

| Category                 | Tools / Libraries                          |
| ------------------------ | ------------------------------------------ |
| **Programming Language** | Python                                     |
| **Libraries**            | pandas, numpy, scikit-learn                |
| **Similarity Metric**    | Cosine Similarity                          |
| **Environment**          | Jupyter Notebook / VS Code                 |
| **Dataset Source**       | TMDB Dataset (Kaggle / The Movie Database) |

---

## 📘 Methodology

### 1. **Data Collection**

* Dataset obtained from **TMDB (The Movie Database)** or **Kaggle**.
* Includes movie titles, genres, overview, keywords, cast, and crew details.

### 2. **Data Preprocessing**

* Handle missing values.
* Combine relevant columns (overview, genres, keywords, etc.) into a single text column.
* Convert text to lowercase and remove unnecessary punctuation.

### 3. **Feature Extraction**

* Use **CountVectorizer** or **TF-IDF Vectorizer** to convert text into numerical feature vectors.

### 4. **Similarity Calculation**

* Apply **Cosine Similarity** to calculate the similarity between movie vectors.

### 5. **Recommendation Generation**

* For a selected movie, sort all movies based on similarity scores and recommend the top 5–10 most similar movies.

---

## 📐 Cosine Similarity Formula

[
\text{Cosine Similarity} = \frac{A \cdot B}{||A|| \times ||B||}
]

Where:

* **A** and **B** are vector representations of two movies.
* The result ranges between **0 (no similarity)** and **1 (perfect similarity)**.

---

## 💻 Implementation Steps

### Step 1: Import Libraries

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### Step 2: Load Dataset

```python
movies = pd.read_csv('tmdb_5000_movies.csv')
```

### Step 3: Combine Features

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']
```

### Step 4: Vectorize and Compute Similarity

```python
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)
```

### Step 5: Define Recommendation Function

```python
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(movies.iloc[i[0]].title)
```

### Step 6: Test the System

```python
recommend('Avatar')
```

✅ **Output Example:**

```
1. Guardians of the Galaxy  
2. Star Trek  
3. Star Wars: The Force Awakens  
4. John Carter  
5. The Fifth Element
```

---

## 📊 Evaluation

Since this is a **content-based system**, evaluation is done using:

* Relevance of the recommended movies.
* Similarity score distribution.
* User satisfaction feedback.

---

## 🚀 Future Enhancements

* Add **Collaborative Filtering** for hybrid recommendations.
* Include **user ratings** and **watch history**.
* Build a **Flask / Streamlit web interface**.
* Deploy the model using **Heroku** or **Render**.

---

## 📂 Project Structure

```
Movie_Recommendation_System/
├── movie_recommendation.ipynb
├── tmdb_5000_movies.csv
├── requirements.txt
├── README.md
└── model/
    └── similarity.pkl
```

---

## 🧾 Dependencies

* pandas
* numpy
* scikit-learn
* nltk *(optional for text cleaning)*
* flask / streamlit *(optional for web app)*

### Install using:

```bash
pip install pandas numpy scikit-learn nltk
```

---

## 👨‍💻 Author

**Akash Chavan**
*Computer Engin
