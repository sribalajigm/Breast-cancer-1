import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ---------------------------------------------------------
# 1. Download UCI Breast Cancer Wisconsin Dataset
# ---------------------------------------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

columns = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

print("Downloading dataset...")

df = pd.read_csv(
    url,
    header=None,
    names=columns
)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# 2. Data preprocessing
# ---------------------------------------------------------

# Convert diagnosis:
# M = 1 (Malignant)
# B = 0 (Benign)

df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"]


# ---------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 4. Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 5. Define Models
# ---------------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=5000),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Naive Bayes":
        GaussianNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )
}


# ---------------------------------------------------------
# 6. Create model directory
# ---------------------------------------------------------

os.makedirs("model", exist_ok=True)


# ---------------------------------------------------------
# 7. Save scaler
# ---------------------------------------------------------

with open("model/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


# ---------------------------------------------------------
# 8. Train and Evaluate Models
# ---------------------------------------------------------

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Probability for AUC
    y_probability = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # Save model
    filename = name.lower().replace(" ", "_") + ".pkl"

    with open(
        f"model/{filename}",
        "wb"
    ) as file:
        pickle.dump(model, file)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n================ RESULTS ================\n")

print(
    results_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 10. Save test data
# ---------------------------------------------------------

test_data = X_test.copy()

test_data["diagnosis"] = y_test.values

test_data.to_csv(
    "test_data.csv",
    index=False
)

print("\nTest data saved as test_data.csv")

print("\nTraining completed successfully!")
