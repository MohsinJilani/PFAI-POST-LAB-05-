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

# Person 2: Logistic Regression model using JAX

import jax
import jax.numpy as jnp
from jax import grad, jit, lax

# Convert existing NumPy data into JAX arrays
X_train = jnp.asarray(x_train)
X_test = jnp.asarray(x_test)
y_train = jnp.asarray(y_train)
y_test = jnp.asarray(y_test)

# Sigmoid function: turns values into probabilities between 0 and 1
def sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))

# Prediction function: applies weights and bias, then passes through sigmoid
def predict(params, X):
    w, b = params
    return sigmoid(X @ w + b)

# Loss function (binary cross-entropy) to measure prediction error
def loss_fn(params, X, y):
    preds = predict(params, X)
    eps = 1e-7  # small value to avoid log(0) errors
    return -jnp.mean(
        y * jnp.log(preds + eps) +
        (1 - y) * jnp.log(1 - preds + eps)
    )

# Compute gradients of the loss function using JAX
loss_grad = grad(loss_fn)

# Person 3: Optimize training and prediction using JIT and lax.fori_loop

import time

# JIT-compiled training step for faster updates
@jit
def train_step(params, X, y, lr):
    grads = loss_grad(params, X, y)
    w, b = params
    dw, db = grads
    new_params = (w - lr * dw, b - lr * db)
    return new_params

# Training function using a JAX loop instead of a Python loop
def train_jax(X, y, lr=0.01, epochs=100):
    n_features = X.shape[1]

    # Initialize weights and bias
    params = (
        jnp.zeros((n_features,), dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32)
    )

    # Run training loop inside JAX for speed
    def body_fn(i, params):
        params = train_step(params, X, y, lr)
        return params

    params = lax.fori_loop(0, epochs, body_fn, params)
    return params

# Warm-up run to compile JAX functions
_ = train_jax(X_train, y_train, lr=0.01, epochs=1)

# Measure training time
start_time = time.perf_counter()
params = train_jax(X_train, y_train, lr=0.01, epochs=100)
jax_train_time = time.perf_counter() - start_time

print("\nFast JAX training completed.")
print("JAX training time:", jax_train_time)

# JIT-compiled batch prediction function
@jit
def predict_batch_jit(params, X):
    return predict(params, X)

# Warm-up prediction to compile
_ = predict_batch_jit(params, X_test).block_until_ready()

# Measure prediction time
start_time = time.perf_counter()
probs = predict_batch_jit(params, X_test)
probs.block_until_ready()
jax_pred_time = time.perf_counter() - start_time

print("JAX batch prediction completed.")
print("JAX prediction time:", jax_pred_time)
print("First 10 predictions:", np.array(probs[:10]))