# Student Performance Prediction System

## 1. Problem Statement
Educational institutions collect student-related data such as attendance, internal marks, study hours, assignment scores, previous academic performance, and other academic factors. However, it can be difficult to identify students who may perform poorly in their final examinations at an early stage.

## 2. Objective
The objective of this project is to develop a Machine Learning-based Student Performance Prediction System that analyzes student-related academic data and predicts their final examination performance. The system also provides useful insights into the factors affecting student performance and helps identify students who may require additional academic support.

## 3. Features
- Exploratory Data Analysis of student attributes.
- Full ML Training Pipeline featuring missing value imputation, one-hot encoding, and feature scaling.
- Evaluation of baseline and advanced regressors (Linear Regression, Decision Trees, Random Forests, Gradient Boosting).
- Interactive Streamlit Dashboard yielding performance predictions on continuous scales.
- Rule-based academic support recommendations.

## 4. Dataset
- **Source:** Kaggle Student Performance Dataset (Mathematics Course)
- **Size:** 395 samples × 33 attributes.
- **Target Variable:** `G3` (Final Examination Score, ranging from 0 - 20)

## 5. Technologies Used
- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Model Persistence:** Joblib

## 6. Machine Learning Algorithms
The project treats this as a regression problem and compares:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor (**Selected Best Model**)
4. Gradient Boosting Regressor

## 7. Data Preprocessing
- Missing values (if any) are evaluated via SimpleImputer.
- Categorical values are dummy-encoded using OneHotEncoder(drop='first').
- Numerical columns are normalized using StandardScaler inside a unified `scikit-learn` Pipeline so that leakage cannot occur between Train/Test boundaries.

## 8. Model Evaluation
The models are evaluated on a strict 20% test partition using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)

## 9. Project Structure
```
student-performance/
│
├── data/
│   └── student-mat.csv
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.pkl
│
├── reports/
│   └── figures/
│
├── src/
│   └── train_model.py
│
├── app.py
├── PROJECT_PRESENTATION_CONTENT.md
├── VIVA_QUESTIONS.md
└── README.md
```

## 10. How to Run the Project

### Installation
Ensure that the correct dependencies are installed. (You typically only need pandas, numpy, scikit-learn, seaborn, and streamlit.)

### Training the Model
To perform EDA, pre-processing, training, and model persistence merely execute:
```bash
python src/train_model.py
```
*This places the saved `.pkl` artifacts into the `models/` folder and charts into `reports/figures`.*

### Running the Web Dashboard (Streamlit)
After training, launch the interactive user interface:
```bash
streamlit run app.py
```
Navigate to your localhost (typically `http://localhost:8501`) via your web browser to test predicting final exam score outcomes.

## 11. Example Workflow
1. Navigate to the Streamlit app.
2. In the "Predict Student Performance" tab sidebar, adjust the student input features like Study Time, Internal Marks (G1/G2), and Absences.
3. Click "Predict Performance".
4. The system outputs the projected score out of 20, classifies the score (e.g. "Good", "Needs Improvement"), and emits an academic advice rationale.
5. In the "Analytics Dashboard" tab, view the dataset metrics, model R² scores, and derived Feature Importances.

## 12. Future Scope
- Incorporating deep learning approaches (like Neural Networks) provided a larger dataset is captured.
- Expanding attributes to cover behavioral and real-time biometric metrics of studying attention.
- Intersecting student data with teacher ratings.
