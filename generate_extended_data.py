import pandas as pd
import numpy as np
import os

print("Generating high-quality synthetic dataset for Final Year Project...")
np.random.seed(42)
n_samples = 5000

# Generating realistic distributions
# Study Hours: Right-skewed slightly (most study 2-5 hours, some study a lot)
study_hours = np.random.gamma(shape=2.5, scale=2.5, size=n_samples).clip(0, 20)

# Attendance: Left-skewed (most students have high attendance)
attendance_base = np.random.beta(a=8, b=2, size=n_samples) * 100

# Previous Score: Normal distribution centered around 70
previous_score = np.random.normal(loc=72, scale=12, size=n_samples).clip(0, 100)

# Sleep Hours: Normal distribution centered around 7.5
sleep_hours = np.random.normal(loc=7.5, scale=1.5, size=n_samples).clip(3, 12)

# Feature interactions & Non-linear target generation
# 1. Study hours have diminishing returns (log or square root)
# 2. Low sleep severely impacts score
# 3. Attendance has a direct multiplier effect
# 4. Previous Score has high correlation

study_impact = np.sqrt(study_hours) * 12
sleep_impact = np.where(sleep_hours < 6, -15 + (sleep_hours - 5)*5, (sleep_hours - 7)*2)
attendance_impact = (attendance_base - 80) * 0.5 
prev_impact = previous_score * 0.4

# Calculate final score with baseline and random noise (unexplainable variance)
noise = np.random.normal(0, 6, n_samples)
final_score_raw = 15 + study_impact + sleep_impact + attendance_impact + prev_impact + noise

# Adding artificial correlation: Students with high previous score tend to study more
study_hours += (previous_score - 70) * 0.05
study_hours = study_hours.clip(0, 20)

# Re-calculate to keep logically consistent
final_score = final_score_raw.clip(0, 100)
attendance = attendance_base.clip(0, 100)

df = pd.DataFrame({
    'Study Hours': study_hours.round(1),
    'Attendance (%)': attendance.round(1),
    'Previous Score': previous_score.round(1),
    'Sleep Hours': sleep_hours.round(1),
    'Final Score': final_score.round(1)
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/student_performance_extended.csv', index=False)
print("Data saved to data/student_performance_extended.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())
