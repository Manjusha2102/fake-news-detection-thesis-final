# Trust-Based Explainable Fake News Detection System

Final thesis project comparing four handcrafted feature sets for fake news detection, evaluated across eight machine learning classifiers with SHAP, LIME, and permutation-based explainability.

---

## Feature Sets

| ID  | Features Used                              |
|-----|--------------------------------------------|
| FS1 | TF-IDF only (baseline)                     |
| FS2 | TF-IDF + Linguistic / writing-style        |
| FS3 | TF-IDF + Emotional / sentiment (VADER + TextBlob) |
| FS4 | TF-IDF + Linguistic + Emotional (full combined) |

**Linguistic features (FS2, FS4):** average sentence length, type-token ratio, modal verb ratio, intensifier ratio, exaggeration ratio, pronoun ratio, capital word ratio, average word length, sentence count, exclamation count, question count.

**Emotional features (FS3, FS4):** VADER positive/negative/neutral/compound scores, TextBlob polarity, TextBlob subjectivity, emotional intensity (|compound|).

---

## Classifiers

Naive Bayes · Decision Tree · Logistic Regression · K-Nearest Neighbour · Linear SVC · Random Forest · Gradient Boosting · XGBoost

---

## Project Structure

```
fake-news-detection-thesis-final/
├── Experiments/
│   ├── Data/
│   │   ├── Dataset_1/          # Kaggle dataset 1 (Fake.csv, True.csv)
│   │   └── Dataset_2/          # Kaggle dataset 2 (Fake.csv, True.csv)
│   ├── fs1_tfidf_only.ipynb
│   ├── fs2_tfidf_linguistic.ipynb
│   ├── fs3_tfidf_emotional.ipynb
│   └── fs4_tfidf_linguistic_emotional.ipynb
└── README.md
```

---

## Datasets

- **Dataset 1:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **Dataset 2:** https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Place the downloaded `Fake.csv` and `True.csv` files in their respective `Data/Dataset_1/` and `Data/Dataset_2/News _dataset/` folders before running.

---

## Requirements

```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost nltk textblob shap lime
```

NLTK corpora downloaded automatically on first run: `stopwords`, `punkt_tab`, `vader_lexicon`, `averaged_perceptron_tagger`.

---

## Running the Experiments

Each notebook is self-contained. Open in Jupyter or Google Colab and run all cells.

**Google Colab:** Update the `FAKE_CSV` / `REAL_CSV` paths at the bottom of each notebook to match your Drive folder structure before running.

**Local:** Uncomment the `else` block at the bottom of each notebook and set the paths to your local CSV files.

---

## Explainability Methods

- **SHAP** — global feature importance (TreeExplainer for tree models, LinearExplainer fallback)
- **LIME** — local explanation for individual article predictions
- **Permutation importance** — measures accuracy drop when each feature is shuffled

---

## Note on Data Leakage

The real-news articles in these datasets contain Reuters dateline patterns (e.g. *"WASHINGTON (Reuters) –"*) that are absent from fake news. Without removal, TF-IDF achieves near-perfect accuracy by memorising these patterns rather than learning linguistic content. All notebooks strip these patterns before feature extraction so results reflect genuine content-based learning.
