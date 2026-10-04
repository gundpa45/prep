import pandas as pd
import numpy as np
import os

np.random.seed(42)
n_samples = 500

study_hours = np.random.normal(5, 2, n_samples).clip(0, 15)
attendance = np.random.normal(85, 10, n_samples).clip(0, 100)
previous_score = np.random.normal(70, 15, n_samples).clip(0, 100)
sleep_hours = np.random.normal(7, 1.5, n_samples).clip(0, 12)

# Create a target variable that logically depends on these
final_score = (
    0.3 * (study_hours / 15 * 100) + 
    0.3 * attendance + 
    0.3 * previous_score + 
    0.1 * (sleep_hours / 12 * 100) +
    np.random.normal(0, 5, n_samples)
).clip(0, 100)

df = pd.DataFrame({
    'Study Hours': study_hours.round(1),
    'Attendance (%)': attendance.round(1),
    'Previous Score': previous_score.round(1),
    'Sleep Hours': sleep_hours.round(1),
    'Final Score': final_score.round(1)
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/simple_student_data.csv', index=False)
print("Saved data/simple_student_data.csv")
