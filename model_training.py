import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("beer_servings.csv")

# Display first rows
print(df.head())

# Dataset info
print(df.info())

# Check null values
print(df.isnull().sum())

# Statistical summary
print(df.describe())

# Correlation
print(df.corr(numeric_only=True))

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Features and target
X = df[['beer_servings', 'spirit_servings', 'wine_servings']]
y = df['total_litres_of_pure_alcohol']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "regression_model.pkl")

print("Model saved successfully!")