#Person 1
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Folder that contains the CSV files
BASE_DIR = r"D:\PFAI\PFAILAB05\PFAI-POST-LAB-05-"
train_path = os.path.join(BASE_DIR, "train.csv")
gender_path = os.path.join(BASE_DIR, "gender_submission.csv")

# Check whether train.csv exists
if not os.path.exists(train_path):
    raise FileNotFoundError(f"train.csv not found at: {train_path}")

# Read the training data
df = pd.read_csv(train_path)

# Read gender_submission.csv if available
if os.path.exists(gender_path):
    gender_df = pd.read_csv(gender_path)
else:
    gender_df = None
    print(f"Warning: gender_submission.csv not found at: {gender_path}")

# Select only useful columns for the model
cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
df = df[cols].copy()

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert categorical columns into numeric form
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Split features and target
x = df.drop("Survived", axis=1).values.astype(np.float32)
y = df["Survived"].values.astype(np.float32)

# Standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x).astype(np.float32)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Print shapes to confirm preprocessing worked
print("Data loaded and preprocessed successfully.")
print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# Preview gender submission file
if gender_df is not None:
    print("\nGender submission preview:")
    print(gender_df.head())