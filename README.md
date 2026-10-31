# IR Project: Comparison of Machine Learning Algorithms for Text Classification

## 1. Project overview
This Information Retrieval project compares three supervised machine-learning algorithms for classifying text documents:

1. K-Nearest Neighbors (KNN)
2. Multinomial Naive Bayes
3. Logistic Regression

The algorithms are evaluated using Accuracy, Precision, Recall, F1-score, training time, and prediction time.

## 2. What is the dataset based on?
The project uses the **20 Newsgroups dataset**. It is not a dataset exclusively about sports or education. It contains real discussion-group messages on many subjects.

For this project, four subject categories are selected:
- `comp.graphics` — computer graphics and computing
- `rec.sport.baseball` — sports, specifically baseball
- `sci.med` — medicine and health-related discussions
- `talk.politics.misc` — politics and public affairs

So the project covers **technology, sports, medicine, and politics**. The model learns to identify which subject category a text message belongs to.

## 3. How the project works
1. Download/load the 20 Newsgroups dataset.
2. Clean the text by removing headers, footers, and quoted messages.
3. Convert text into numerical TF-IDF features.
4. Split the data into 75% training and 25% testing data.
5. Train the three algorithms.
6. Measure and compare their performance.
7. Rank the algorithms by weighted F1-score.
8. Save comparison tables, classification reports, confusion matrices, and a graph in `results/`.

## 4. Installation
Install Python 3.9 or later. Open a terminal in this project folder and run:

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Run the project

```bash
python main.py
```

On the first run, scikit-learn downloads the 20 Newsgroups dataset. Internet access is needed for this first download. Later runs use the local dataset cache.

## 6. Output files
The program creates these files inside `results/`:
- `model_comparison.csv` — metrics and final ranking
- `algorithm_comparison.png` — comparison graph
- `*_confusion_matrix.png` — confusion matrix for each algorithm
- `*_classification_report.txt` — precision, recall, and F1-score by category
- `dataset_summary.txt` — dataset and split details

The exact scores are generated when you run the code. They are not hard-coded because they depend on the dataset version and execution environment.

## 7. Algorithms in simple words
- **KNN:** predicts a document's category by looking at the categories of the most similar training documents.
- **Multinomial Naive Bayes:** uses word frequencies and probabilities to estimate the most likely category.
- **Logistic Regression:** learns how strongly words indicate each category and predicts the most likely one.

## 8. Evaluation metrics
- **Accuracy:** proportion of all test documents classified correctly.
- **Precision:** among documents predicted as a category, how many truly belong to it.
- **Recall:** among documents truly belonging to a category, how many were found.
- **F1-score:** balance between precision and recall.
- **Training time:** time taken to train the model.
- **Prediction time:** time taken to classify the test documents.

## 9. Important note
Do not write that one algorithm is the winner until you run the program and inspect `results/model_comparison.csv`. The program ranks models automatically by weighted F1-score.

## 10. Suggested GitHub repository name
`IR-ML-Algorithm-Comparison`
# IR-TASK-2
