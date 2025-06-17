# 🎬 IMDB Movie Review Sentiment Analysis with Logistic Regression

Sentiment analysis on IMDB movie reviews using Python, NLP, and Logistic Regression in an end-to-end ML pipeline.

---

## 📌 Project Objective

To build a robust and accurate machine learning model that:
- Understands and processes raw textual movie reviews
- Classifies them into **Positive** or **Negative** sentiment
- Can generalize to unseen reviews with high confidence

---

## 📂 Dataset

- **Name**: [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Total reviews**: 50,000  
  - 25,000 positive  
  - 25,000 negative  
- Balanced dataset with binary sentiment labels  
- Used in `.csv` format with `review` and `sentiment` columns

---

## 🛠️ Technologies Used

| Tool/Library         | Purpose                                  |
|----------------------|------------------------------------------|
| Python               | Programming language                     |
| Pandas               | Data loading & manipulation              |
| NumPy                | Numerical operations                     |
| Matplotlib & Seaborn | Data visualization                       |
| NLTK                 | Natural language preprocessing           |
| Scikit-learn         | Machine Learning pipeline & evaluation   |
| TfidfVectorizer      | Feature extraction from text             |
| LogisticRegression   | Classification model                     |

---

## 🧠 Methodology

1. **Data Cleaning**  
   - Convert text to lowercase  
   - Remove non-alphabetic characters  
   - Tokenize text  
   - Remove stopwords using NLTK  
   - Apply stemming using `PorterStemmer`

2. **Exploratory Data Analysis (EDA)**  
   - Checked dataset structure  
   - Plotted sentiment distribution

3. **Text Vectorization**  
   - Applied `TfidfVectorizer` with max 5000 features

4. **Model Building**  
   - Used `Logistic Regression` within a Scikit-learn Pipeline  
   - Train-test split (80/20)

5. **Evaluation**  
   - Classification report  
   - Confusion matrix

6. **Prediction**  
   - Tested the model on new reviews

---

## 📈 Results

### 🔹 Classification Report

**Accuracy: 89%**

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Negative (0) | 0.90      | 0.87   | 0.88     | 4961    |
| Positive (1) | 0.87      | 0.90   | 0.89     | 5039    |
| **Overall**  |           |        | **0.89** | 10000   |

✅ The model performs with high accuracy and balanced class metrics.

---

### 🔹 Graph 1: Sentiment Distribution

Displays the equal distribution of positive and negative reviews.

![Sentiment Distribution](samples/sentiment_distribution.png)

---

### 🔹 Graph 2: Confusion Matrix

Shows predicted vs actual sentiment labels. A strong diagonal reflects correct classifications.

![Confusion Matrix](samples/confusion_matrix.png)

---

## 🧪 Live Predictions

Tested on two custom reviews:

```python
new_reviews = [
    "I absolutely loved this movie! The storyline was fantastic and the acting was superb.",
    "This was the worst film I have ever seen. It was a total waste of time."
]
```

## 🚀 How to Run This Project

# 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

# 2. (Optional) Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # For Windows
# Or use: source venv/bin/activate  # For Linux/Mac
```

# 3. Install the required dependencies
```bash
pip install -r requirements.txt
```

# 4. Run the Python script
```bash
python Sentiment_Analysis.py
```

