import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

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


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🧬 Breast Cancer Classification System")

st.write(
    "Machine Learning classification using multiple models "
    "on the UCI Breast Cancer Wisconsin Diagnostic Dataset."
)


# =========================================================
# MODEL DIRECTORY
# =========================================================

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model folder
MODEL_DIR = os.path.join(BASE_DIR, "model")


# =========================================================
# CHECK MODEL DIRECTORY
# =========================================================

if not os.path.exists(MODEL_DIR):

    st.error(
        "❌ The 'model' folder was not found.\n\n"
        f"Expected location: {MODEL_DIR}"
    )

    st.stop()


# =========================================================
# REQUIRED MODEL FILES
# =========================================================

required_files = [
    "decision_tree.pkl",
    "knn.pkl",
    "logistic_regression.pkl",
    "naive_bayes.pkl",
    "random_forest.pkl",
    "scaler.pkl",
    "svm.pkl"
]


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

missing_files = []

for filename in required_files:

    file_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.isfile(file_path):

        missing_files.append(filename)


if missing_files:

    st.error(
        "❌ The following required model files are missing:"
    )

    for filename in missing_files:

        st.write(f"- {filename}")

    st.write(
        "Make sure all .pkl files are inside the "
        "`model` folder in your GitHub repository."
    )

    st.stop()


# =========================================================
# LOAD SCALER
# =========================================================

scaler_path = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

try:

    with open(
        scaler_path,
        "rb"
    ) as file:

        scaler = pickle.load(file)

except Exception as e:

    st.error(
        f"❌ Error loading scaler.pkl: {e}"
    )

    st.stop()


# =========================================================
# MODEL DICTIONARY
# =========================================================

model_files = {

    "Logistic Regression":
        os.path.join(
            MODEL_DIR,
            "logistic_regression.pkl"
        ),

    "Decision Tree":
        os.path.join(
            MODEL_DIR,
            "decision_tree.pkl"
        ),

    "KNN":
        os.path.join(
            MODEL_DIR,
            "knn.pkl"
        ),

    "Naive Bayes":
        os.path.join(
            MODEL_DIR,
            "naive_bayes.pkl"
        ),

    "Random Forest":
        os.path.join(
            MODEL_DIR,
            "random_forest.pkl"
        ),

    "SVM":
        os.path.join(
            MODEL_DIR,
            "svm.pkl"
        )
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(model_files.keys())
)


# =========================================================
# LOAD SELECTED MODEL
# =========================================================

selected_model_path = model_files[selected_model]


# Extra safety check
if not os.path.isfile(selected_model_path):

    st.error(
        "❌ Selected model file was not found."
    )

    st.write(
        f"Expected file: `{selected_model_path}`"
    )

    st.stop()


try:

    with open(
        selected_model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

except Exception as e:

    st.error(
        f"❌ Error loading {selected_model}.pkl"
    )

    st.write(str(e))

    st.stop()


# =========================================================
# DISPLAY SELECTED MODEL
# =========================================================

st.subheader(
    f"Selected Model: {selected_model}"
)


# =========================================================
# DATASET UPLOAD
# =========================================================

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# =========================================================
# PROCESS DATA
# =========================================================

if uploaded_file is not None:

    try:

        data = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"❌ Error reading CSV file: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # DISPLAY DATASET
    # -----------------------------------------------------

    st.write("Uploaded Dataset")

    st.dataframe(
        data.head()
    )


    # -----------------------------------------------------
    # CHECK TARGET COLUMN
    # -----------------------------------------------------

    if "diagnosis" not in data.columns:

        st.error(
            "❌ The uploaded CSV must contain "
            "a 'diagnosis' column."
        )

        st.stop()


    # -----------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # -----------------------------------------------------

    X = data.drop(
        columns=["diagnosis"]
    )

    y = data["diagnosis"]


    # -----------------------------------------------------
    # CHECK FEATURE COUNT
    # -----------------------------------------------------

    if hasattr(scaler, "n_features_in_"):

        expected_features = scaler.n_features_in_

        actual_features = X.shape[1]

        if actual_features != expected_features:

            st.error(
                f"❌ Incorrect number of features.\n\n"
                f"Model expects {expected_features} features, "
                f"but uploaded dataset contains "
                f"{actual_features} features."
            )

            st.stop()


    # -----------------------------------------------------
    # SCALE FEATURES
    # -----------------------------------------------------

    try:

        X_scaled = scaler.transform(
            X
        )

    except Exception as e:

        st.error(
            f"❌ Error while scaling the input data: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        predictions = model.predict(
            X_scaled
        )

    except Exception as e:

        st.error(
            f"❌ Error while making predictions: {e}"
        )

        st.stop()


    # -----------------------------------------------------
    # PREDICTION PROBABILITY
    # -----------------------------------------------------

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = model.predict_proba(
                X_scaled
            )[:, 1]

        except Exception:

            probabilities = None


    # =====================================================
    # METRICS
    # =====================================================

    accuracy = accuracy_score(
        y,
        predictions
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
    # AUC
    # -----------------------------------------------------

    auc = None

    if probabilities is not None:

        try:

            auc = roc_auc_score(
                y,
                probabilities
            )

        except Exception:

            auc = None


    # =====================================================
    # DISPLAY METRICS
    # =====================================================

    st.subheader(
        "Evaluation Metrics"
    )

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

        if auc is not None:

            st.metric(
                "AUC",
                f"{auc:.4f}"
            )

        else:

            st.metric(
                "AUC",
                "N/A"
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


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y,
        predictions
    )


    fig, ax = plt.subplots()

    ax.imshow(
        cm
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )


    for i in range(
        len(cm)
    ):

        for j in range(
            len(cm[i])
        ):

            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center"
            )


    st.pyplot(
        fig
    )

    plt.close(fig)


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

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


    # =====================================================
    # PREDICTION RESULTS
    # =====================================================

    st.subheader(
        "Prediction Results"
    )


    result = data.copy()


    result["Predicted Diagnosis"] = predictions


    if probabilities is not None:

        result[
            "Prediction Probability"
        ] = probabilities


    st.dataframe(
        result
    )


else:

    st.info(
        "Please upload test_data.csv "
        "to evaluate the selected model."
    )
```
