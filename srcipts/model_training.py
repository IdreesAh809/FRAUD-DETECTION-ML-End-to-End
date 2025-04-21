import pandas as pd
import numpy as np
import os
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
from xgboost import XGBClassifier
from cuml.neighbors import KNeighborsClassifier

df = pd.read_csv("d:\\Volume E\\project\\FRAUD-DETECTION-ML\\data\\processed\\feature_engineered_data.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])

# Save the new split datasets
print("\nTrain and Test datasets saved successfully!")

X_train, y_train = train.drop(columns=['Class']), train['Class']
X_test, y_test = test.drop(columns=['Class']), test['Class']

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(tree_method='hist', device='cuda'),
    "KNN": KNeighborsClassifier()
}




# *Step 3: Training and Saving Models*
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Save each trained model
    model_filename = f"{models_dir}/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_filename)
    print(f"{name} model saved successfully!")