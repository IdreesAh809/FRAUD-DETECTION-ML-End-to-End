from google.colab import drive
drive.mount('/content/drive')

import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define paths for models and datasets
models_dir = "/content/drive/MyDrive/models"
test_data_path = "/content/drive/MyDrive/test_data.csv"
train_data_path = "/content/drive/MyDrive/train_data.csv"

# Load test and train data
test_df = pd.read_csv(test_data_path)
train_df = pd.read_csv(train_data_path)

# Extract features and target variable
X_test = test_df.drop(columns=['Class'])  # Adjust 'Class' column name if different
y_test = test_df['Class']
X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']

# Dictionary containing model names and their corresponding file paths
model_filenames = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "Support Vector Machine": "svm.pkl",
    "XGBoost": "xgboost.pkl",
    "K-Nearest Neighbors": "knn.pkl"
}

# Initialize list to store evaluation results
results = []


# Iterate through models, load them, and evaluate
for model_name, filename in model_filenames.items():
    model_path = os.path.join(models_dir, filename)

    if os.path.exists(model_path):
        # Load the saved model
        model = joblib.load(model_path)

        # Predict on train and test datasets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute evaluation metrics
        metrics = {
            'Model': model_name,
            'Train Accuracy': accuracy_score(y_train, y_train_pred),
            'Test Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred, zero_division=1),
            'Recall': recall_score(y_test, y_test_pred, zero_division=1),
            'F1 Score': f1_score(y_test, y_test_pred, zero_division=1)
        }
        results.append(metrics)
    else:
        print(f"Model file not found: {model_path}")

# Convert results to DataFrame and display
evaluation_df = pd.DataFrame(results)
print(evaluation_df)

# Save evaluation results to CSV