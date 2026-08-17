# Machine Learning Assignment 2

## a. Problem Statement

The objective of this project is to develop and compare multiple
machine learning classification models for predicting whether a
breast tumor is benign or malignant.

The project also demonstrates the deployment of the trained models
through an interactive Streamlit web application.

---

## b. Dataset Description

Dataset:
Breast Cancer Wisconsin Diagnostic Dataset

Source:
UCI Machine Learning Repository

Problem Type:
Binary Classification

Number of Instances:
569

Number of Features:
30

Target Variable:
Diagnosis

Classes:
B - Benign
M - Malignant

The dataset satisfies the minimum requirements of 500 instances
and 12 features specified in the assignment.

---

## c. GitHub Repository Link

Paste your GitHub repository link here.

Example:

https://github.com/YOUR_USERNAME/ML_Assignment_2

---

## d. Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier
6. Support Vector Machine

---

## Evaluation Metrics

The following evaluation metrics were calculated:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest | | | | | | |
| SVM | | | | | | |

---

## Model Performance Observations

### Logistic Regression

Logistic Regression provides a strong baseline for binary
classification and performs well when the feature relationships
are reasonably linear.

### Decision Tree

The Decision Tree is easy to interpret and can model nonlinear
relationships. However, it can be more sensitive to overfitting.

### KNN

KNN classifies observations based on neighboring samples.
Feature scaling is important for obtaining reliable performance.

### Naive Bayes

Gaussian Naive Bayes is computationally efficient and provides a
simple probabilistic classification approach.

### Random Forest

Random Forest combines multiple decision trees and generally
provides robust performance while reducing the overfitting
associated with an individual decision tree.

### SVM

SVM is effective for classification problems with high-dimensional
feature spaces and can produce strong decision boundaries after
feature scaling.

---

## Overall Winner

The overall winner is the model that achieves the best balance
across Accuracy, AUC, Precision, Recall, F1 Score and MCC on the
test dataset.

The winner will be selected based on the actual results generated
during model evaluation.

---

## Streamlit Application

The Streamlit application provides:

- Test CSV upload
- Model selection
- Prediction
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion matrix
- Classification report
- Prediction results

---

## Streamlit App Link

Paste your deployed Streamlit URL here.

---

## Project Structure

ML_Assignment_2/

├── app.py

├── train_models.py

├── requirements.txt

├── README.md

├── test_data.csv

└── model/

    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── svm.pkl
    └── scaler.pkl