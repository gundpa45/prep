import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

os.makedirs('models', exist_ok=True)

print("Loading data...")
df = pd.read_csv('data/simple_student_data.csv')

X = df[['Study Hours', 'Attendance (%)', 'Previous Score', 'Sleep Hours']]
y = df['Final Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = Ridge()
model.fit(X_train, y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)

print(f"Model trained! R^2 Score: {r2:.4f}")

joblib.dump(model, 'models/simple_model.joblib')
joblib.dump(r2, 'models/simple_model_r2.joblib')
print("Model saved to models/simple_model.joblib")
