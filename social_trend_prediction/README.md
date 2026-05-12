# Social Media Analytics for Trend Prediction Using Machine Learning

## Student Details
Name: Sneha Pal  
Roll Number: 2210990996  

## Project Details
Project Title: Social Media Analytics for Trend Prediction Using Machine Learning  
Project Type: Research  
Department: Computer Science and Engineering  
University: Chitkara University  

## Team Details
Team Member 1: Sneha Pal - 2210990996  


## Submission Status
Submitted


# Social Media Analytics for Trend Prediction Using Machine Learning

> Companion code for the research paper by **Sneha Pal**  
> Department of Computer Science and Engineering, Chitkara University, Punjab, India

---

## 📋 Overview

This project implements an end-to-end Machine Learning pipeline for predicting social media trends — reproducing all results and figures from the paper.

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 82.3%    | 80.1%     | 78.6%  | 79.3%    |
| Naive Bayes         | 76.8%    | 74.5%     | 73.2%  | 73.8%    |
| **Random Forest**   | **91.0%**| **89.5%** |**88.7%**|**89.1%**|

The Random Forest model detects trends approximately **4.2 hours** before they officially appear on platform trending lists.

---

## 📁 Project Structure

```
social_trend_prediction/
│
├── main.py                  # Full ML pipeline — runs everything
├── generate_dataset.py      # Synthetic tweet CSV generator
├── requirements.txt         # Python dependencies
│
├── data/
│   └── tweets_sample.csv    # Auto-generated (150,000 tweets)
│
└── outputs/
    ├── accuracy_chart.png   # Fig 1 — Test-set accuracy comparison
    ├── metrics_bar.png      # Fig 2 — Precision / Recall / F1 bar chart
    ├── feature_importance.png # Fig 3 — Top-10 Random Forest features
    └── trend_timeseries.png # Fig 4 — 72-hour trend probability curve
```

---

## ⚙️ Installation

**Python 3.10+ required.**

```bash
# Clone / navigate into the project folder
cd social_trend_prediction

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1 — Generate the tweet dataset (if you don't have Twitter API access)

```bash
python generate_dataset.py
```

This creates `data/tweets_sample.csv` with **150,000 synthetic tweets** matching the paper's distribution:
- 31% trending, 69% non-trending
- Columns: `timestamp`, `text`, `hashtags`, `like_count`, `retweet_count`, `hashtag_count`, `hashtag_24h_freq`, `hashtag_velocity`, `sentiment_polarity`, `retweet_like_ratio`, `trending`

### Step 2 — Run the full pipeline

```bash
python main.py
```

This will:
1. Load and preprocess the tweet data (URL removal, lowercasing, stop-word filtering, lemmatisation)
2. Extract TF-IDF features (up to 10,000 unigrams + bigrams) + engagement/hashtag features
3. Train Logistic Regression, Naive Bayes, and Random Forest classifiers
4. Generate all 4 output charts matching the paper's figures

---

## 📊 Output Figures

| File | Paper Figure | Description |
|------|-------------|-------------|
| `accuracy_chart.png` | Fig 1 | Bar chart comparing test-set accuracy across 3 models |
| `metrics_bar.png` | Fig 2 | Grouped bar chart — Precision, Recall, F1 per model |
| `feature_importance.png` | Fig 3 | Top-10 Random Forest feature importances |
| `trend_timeseries.png` | Fig 4 | Predicted trend probability vs. actual status over 72 hours |

---

## 🧠 Methodology (Paper Summary)

### Preprocessing Pipeline
- Remove URLs (`http://...`) and mentions (`@user`)
- Lowercase normalisation
- Punctuation stripping
- NLTK stop-word removal
- WordNet lemmatisation

### Feature Engineering

| Feature Group | Details |
|---------------|---------|
| **TF-IDF** | Max 10,000 unigrams + bigrams; IDF-weighted |
| **Hashtag velocity** | Unique count, 24-h frequency, rate-of-change |
| **Engagement metrics** | Like count, retweet count, retweet-to-like ratio (min-max normalised) |
| **Sentiment polarity** | VADER-style compound score |

### Models
- **Logistic Regression** — L2 regularised, C=1.0, class-weight balanced
- **Multinomial Naive Bayes** — alpha=1.0
- **Random Forest** — 200 trees, class-weight balanced, 5-fold CV grid search

---

## 📝 About the Dataset

The paper used ~150,000 Twitter posts collected via the **Academic Research API** over 6 months, across 5 domains: technology, entertainment, politics, sports, and public health.

Since the Twitter API is now restricted, this repo includes `generate_dataset.py` which creates a **realistic synthetic dataset** with the same statistical properties:

- Same class ratio (31%/69%)
- Realistic engagement distributions (trending vs. non-trending)
- Hashtag velocity and frequency patterns
- Sentiment polarity ranges

### To use real Twitter data instead:
Replace `data/tweets_sample.csv` with your own CSV containing these columns:

```
timestamp, text, hashtags, like_count, retweet_count,
hashtag_count, hashtag_24h_freq, hashtag_velocity,
sentiment_polarity, retweet_like_ratio, trending
```

The `trending` column should be `1` (trending) or `0` (not trending).

---

## 📚 Key References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Hutto, C. J. & Gilbert, E. (2014). VADER: Sentiment analysis for social media. *AAAI ICWSM*.
- Salton, G. & Buckley, C. (1988). Term-weighting in automatic text retrieval. *Information Processing & Management*.
- Devlin et al. (2019). BERT. *NAACL-HLT*.

---

## 👤 Author

**Sneha Pal** — B.E. Computer Science and Engineering, Chitkara University Institute of Engineering and Technology, Punjab, India  
sneh996.be22@chitkara.edu.in
