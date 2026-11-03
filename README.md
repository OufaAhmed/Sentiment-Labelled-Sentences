# 📊 Sentiment Analysis using Naive Bayes

An end-to-end Machine Learning project that performs Sentiment Analysis on combined text datasets (Amazon, IMDb, and Yelp) using Python and Scikit-Learn. The goal is to automatically classify text sentences as either **Positive (1)** or **Negative (0)**.

---

## 🚀 Project Overview
This project takes raw textual reviews, cleans them using Natural Language Processing (NLP) techniques, converts them into numerical vectors, and trains a **Multinomial Naive Bayes** classifier to predict sentiments accurately.

---

## 🛠️ Tech Stack & Libraries
* **Python** (Core programming language)
* **Pandas** (Data manipulation and handling DataFrames)
* **Scikit-Learn (`sklearn`)**:
  * `train_test_split` (For stratified data splitting)
  * `CountVectorizer` (For text vectorization and feature extraction)
  * `MultinomialNB` (The core Naive Bayes classifier model)
  * Evaluation Metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, `confusion_matrix`)
* **Regular Expressions (`re`)** (For text cleaning and preprocessing)

---

## 🔄 Project Pipeline (Step-by-Step)
1. **Data Loading & Combination:** Loaded multiple review datasets (Amazon, IMDb, Yelp) and combined them into a single unified DataFrame.
2. **Data Cleaning:** 
   * Dropped missing/null values (`dropna`).
   * Removed duplicate sentences to avoid model bias (`drop_duplicates`).
   * Cleaned text using RegEx (`re.sub`) to remove unwanted characters and standardized text format using `.strip()`.
3. **Stratified Train-Test Split:** Split the dataset into 80% training and 20% testing data using `stratify=y` to ensure an equal and fair class distribution.
4. **Text Vectorization:** Converted text data into numerical token counts using `CountVectorizer` (`fit_transform` on training data, `transform` on testing data).
5. **Model Training:** Trained a `MultinomialNB` classifier with Laplace smoothing (`alpha=1.0`).
6. **Evaluation & Prediction:** Evaluated model performance using rigorous metrics (Accuracy, Precision, Recall, F1-score, and Confusion Matrix).

---

## 📈 Model Performance & Results

* **Overall Test Accuracy:** **~79.5%**

### Classification Report (Positive vs. Negative Classes)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (0)** | 0.79 | 0.80 | 0.79 | 271 |
| **Positive (1)** | 0.80 | 0.79 | 0.80 | 276 |

* **Key Insights:**
  * The model demonstrates a **highly balanced performance** across both sentiment classes with an overall accuracy of ~80%.
  * F1-scores ($\approx 0.79 - 0.80$) confirm that the model generalizes robustly without leaning unfairly toward either class.

---

## 💻 How to Run the Code
1. Clone this repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
