import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🧬 Breast Cancer Classification System")

st.write(
    "Machine Learning classification using multiple models "
    "on the UCI Breast Cancer Wisconsin Diagnostic Dataset."
)


# ---------------------------------------------------------
# Load scaler
# ---------------------------------------------------------

with open("model/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
with open("model/logistic_regression.pkl", "rb") as file:
    model = pickle.load(file)


# ---------------------------------------------------------
# Model dictionary
# ---------------------------------------------------------

model_files = {

    "Logistic Regression":
        "model/logistic_regression.pkl",

    "Decision Tree":
        "model/decision_tree.pkl",

    "KNN":
        "model/knn.pkl",

    "Naive Bayes":
        "model/naive_bayes.pkl",

    "Random Forest":
        "model/random_forest.pkl",

    "SVM":
        "model/svm.pkl"
}


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(model_files.keys())
)


# ---------------------------------------------------------
# Load selected model
# ---------------------------------------------------------

with open(
    model_files[selected_model],
    "rb"
) as file:

    model = pickle.load(file)


st.subheader(
    f"Selected Model: {selected_model}"
)


# ---------------------------------------------------------
# Dataset Upload
# ---------------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(
        uploaded_file
    )

    st.write("Uploaded Dataset")

    st.dataframe(
        data.head()
    )


    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    if "diagnosis" not in data.columns:

        st.error(
            "The uploaded CSV must contain a "
            "'diagnosis' column."
        )

        st.stop()


    X = data.drop(
        columns=["diagnosis"]
    )

    y = data["diagnosis"]


    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    X_scaled = scaler.transform(X)


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predictions = model.predict(
        X_scaled
    )

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y,
        predictions
    )


    # -----------------------------------------------------
    # Display Metrics
    # -----------------------------------------------------

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        st.metric(
            "Precision",
            f"{precision:.4f}"
        )


    with col2:

        st.metric(
            "AUC",
            f"{auc:.4f}"
        )

        st.metric(
            "Recall",
            f"{recall:.4f}"
        )


    with col3:

        st.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        st.metric(
            "MCC",
            f"{mcc:.4f}"
        )


    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y,
        predictions
    )

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    for i in range(len(cm)):

        for j in range(len(cm[i])):

            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)


    # -----------------------------------------------------
    # Classification Report
    # -----------------------------------------------------

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y,
        predictions,
        target_names=[
            "Benign",
            "Malignant"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df
    )


    # -----------------------------------------------------
    # Prediction Results
    # -----------------------------------------------------

    st.subheader(
        "Prediction Results"
    )

    result = data.copy()

    result["Predicted Diagnosis"] = predictions

    result["Prediction Probability"] = probabilities

    st.dataframe(
        result
    )

else:

    st.info(
        "Please upload test_data.csv to evaluate the selected model."
    )
