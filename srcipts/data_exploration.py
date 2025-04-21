import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv(r"d:\Volume E\project\FRAUD-DETECTION-ML\data\raw\creditcard.csv")

# Step 2: Basic exploration
print("Dataset Shape:", df.shape)
print("\nColumn Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nClass Distribution:")
print(df['Class'].value_counts())  # Updated column name

# Step 3: Summary Statistics
print("\nSummary Statistics:")
print(df.describe())


# Step 4: Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Remove duplicates
df = df.drop_duplicates()
print("\nDataset Shape after removing duplicates:", df.shape)

# Step 5: Data Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
# Feature Distribution
df.drop(columns=['Time', 'Amount', 'Class']).hist(figsize=(15, 12), bins=30, color='blue', alpha=0.7)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()
# Box Plot for Fraud Detection
plt.figure(figsize=(12, 6))
sns.boxplot(x='Class', y='Amount', data=df)
plt.yscale('log')  # Log scale for better visualization
plt.title("Transaction Amounts for Fraud vs Non-Fraud")
plt.show()

# Step 6: Train-Test Split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])


# Step 7: Ensure output directory exists
os.makedirs("../data/processed", exist_ok=True)

# Step 8: Save processed datasets

print("\nTrain, Test, and Cleaned datasets saved successfully!")
