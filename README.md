# Fake News Detection System Using Machine Learning

## 📌 Project Overview

The **Fake News Detection System** is a Machine Learning–based project designed to automatically classify news articles as **Real** or **Fake**. With the rapid spread of misinformation on digital platforms, manual fact-checking is no longer scalable. This project demonstrates how **Natural Language Processing (NLP)** and **Machine Learning (ML)** can be used to build an efficient and accurate fake news classifier.

This project is suitable for:

* Academic projects & internships
* Machine Learning / Data Science portfolios
* Understanding NLP-based text classification

---

## 🎯 Objectives

* Detect fake and real news automatically
* Apply NLP techniques for text preprocessing
* Use TF-IDF for feature extraction
* Train and evaluate a Machine Learning model
* Provide clear performance metrics

---

## 🧠 Technologies & Tools Used

* **Programming Language:** Python
* **Libraries:**

  * Pandas
  * NumPy
  * Scikit-learn
  * NLTK
  * Matplotlib / Seaborn
  * WordCloud
* **IDE/Environment:** Jupyter Notebook

---

## 📂 Dataset Information

* **Dataset Name:** Fake News Detection Dataset
* **Source:** Kaggle
* **Total Samples:** 40,587 news articles
* **Features:**

  * `title` – Headline of the news
  * `text` – Full news content
  * `label` – 1 (Real), 0 (Fake)

---

## ⚙️ System Architecture

```
Raw News Data
      ↓
Text Preprocessing
      ↓
Feature Extraction (TF-IDF)
      ↓
Model Training (Logistic Regression)
      ↓
Model Evaluation
      ↓
Prediction (Real / Fake)
```

---

## 🔍 Methodology

### 1. Data Preprocessing

* Lowercasing text
* Removing punctuation and numbers
* Stopword removal
* Stemming / Lemmatization

### 2. Feature Engineering

* TF-IDF Vectorization to convert text into numerical features

### 3. Model Building

* Logistic Regression classifier used for binary classification

### 4. Model Evaluation

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📊 Results

The Logistic Regression model provides:

* High accuracy on test data
* Balanced precision and recall
* Efficient performance on large textual datasets

(Exact metrics may vary based on random state and preprocessing)

---

## 📈 Visualizations

* Class distribution plot
* WordCloud for Real news
* WordCloud for Fake news
* Confusion Matrix heatmap

---

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git
```

2. Navigate to the project directory:

```bash
cd fake-news-detection
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Open Jupyter Notebook:

```bash
jupyter notebook
```

5. Run the notebook:

```
Fake_News_Detection_System_Using_Machine_Learning.ipynb
```

---

## 📌 Future Enhancements

* Use advanced models like:

  * Naive Bayes
  * Random Forest
  * XGBoost
  * LSTM / BERT
* Deploy model using Flask or Streamlit
* Add real-time news URL prediction
* Multilingual fake news detection

---

## 👩‍💻 Author

**Sakshi**
Aspiring Data Scientist | Machine Learning Enthusiast

---

## ⭐ Acknowledgements

* Kaggle for providing the dataset
* Scikit-learn & NLTK documentation
* Open-source community

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify it.

---

> ⭐ If you found this project helpful, don’t forget to give it a **star** on GitHub!
