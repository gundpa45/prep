# Student Performance Prediction System: Presentation Content

## Slide 1: Title Slide
- **Title**: Student Performance Prediction System Using Machine Learning
- **Subtitle**: Final Year ML Mini Project
- **Presenter**: [Your Name/Team]
- **Major**: Computer Engineering
- **Verbal Notes**: "Good morning respected professors and jury members. Today, I am presenting my final-year mini project on identifying and predicting student academic performance early using Machine Learning."

## Slide 2: Introduction
- Education collects vast amounts of data beyond just grades.
- Data encompasses lifestyle, sociodemographic factors, and attendance.
- Proactive identification is highly limited currently.
- Early intervention is necessary for academic success.
- **Verbal Notes**: "Schools collect continuous data ranging from attendance to internal assessments. However, identifying at-risk students happens too late. Our goal is to leverage this data proactively."

## Slide 3: Problem Statement
- Difficult to identify academically struggling students early via plain rules alone.
- Delayed interventions lead to high failure rates.
- Need an automated, data-driven system to predict the final output.
- **Verbal Notes**: "The core problem is that manual tracking of all student demographics, lifestyle habits, and intermittent marks is incredibly difficult for teachers. We aim to automate tracking so no student is left behind."

## Slide 4: Objectives
- Develop a Machine Learning Regression Model to forecast final scores.
- Uncover major factors influencing exam performance using data science.
- Provide a responsive Web Dashboard for quick access by educators.
- Generate automated, rule-based recommendations for students at varying risk tiers.
- **Verbal Notes**: "Our objectives are multi-fold: we don't just want to blindly predict a number; we want to extract analytical insights and feature importances, bundled into an easy-to-use Streamlit dashboard."

## Slide 5: Dataset
- **Source**: Kaggle Student Performance Dataset (Mathematics Course).
- **Size**: 395 samples, 33 attributes.
- **Features Include**: Demographics, Study time, Absences, Past Failures, Internal marks (G1, G2).
- **Target Variable**: 'G3' (Final examination score out of 20).
- **Verbal Notes**: "We utilized an authentic UCI dataset from Kaggle comprising 395 students. It has 33 attributes. We designate the 'G3' variable, which is the final exam score out of 20, as our regression target."

## Slide 6: Data Preprocessing
- **Missing Values**: Checked via SimpleImputer.
- **Categorical Columns (17)**: Converted to numerical using OneHotEncoder.
- **Numerical Columns (15)**: Scaled via StandardScaler to handle diverse unit distributions.
- **Leakage Prevention**: All transformations applied via pipeline after an 80/20 train/test split.
- **Verbal Notes**: "Before feeding data to an algorithm, it must be clean in a strictly reproducible manner. We utilized Scikit-learn Pipelines with OneHotEncoding and Scaling, fitted purely on the training set to prevent strictly forbidden data leakage."

## Slide 7: Exploratory Data Analysis (EDA)
- Strong linear correlation discovered between internal assessments (G1, G2) and Final Score.
- Higher absences trend towards a mildly lower final outcome.
- Study-time has a positive boxplot influence on median grades.
- Past class failures heavily deprecate the final final score.
- **Verbal Notes**: "During EDA, we visualized critical insights. Non-surprisingly, internal assessments correlate almost perfectly with final grades, while social features like extremely high absences subtly degrade performance."

## Slide 8: Machine Learning Algorithms
- **Treated as a Regression Task** since 'G3' is continuous numeric.
- **Linear Regression**: Baseline reference algorithm.
- **Decision Tree**: Captures nonlinear interactions.
- **Random Forest**: Ensemble bagging approach, controls variance and overfitting.
- **Gradient Boosting**: Ensemble boosting method improving via sequential error reduction.
- **Verbal Notes**: "We approached this primarily as a regression problem. We compared four algorithms ranging from baseline linear methods to complex ensemble trees to ensure robust predictive capability regardless of feature structure."

## Slide 9: Model Evaluation & Comparison
- Evaluated on Unseen 20% test data partition.
- Linear Regression R²: ~0.72
- Random Forest Regressor R²: ~0.81 (Best Performer)
- Gradient Boosting demonstrated near identical success (~0.81)
- Validation metric emphasis on Root Mean Squared Error (RMSE).
- **Verbal Notes**: "When evaluated against the held-out test set, the Random Forest Regressor proved the most stable and performant, achieving an R² accuracy score near 81% and consistently the lowest Mean Absolute Error."

## Slide 10: Streamlit Application
- Created utilizing Python's `streamlit` library.
- Model persistence via `Joblib`. (Never retrains model on each page load).
- Dynamic inputs utilizing sliders and human-readable select boxes.
- Dual interface: Predictor Page & Analytics Dashboard.
- **Verbal Notes**: "The outcome of this pipeline is deployed actively inside a Streamlit web application. We've persisted the exact pipeline so the dashboard can evaluate real-world inputs instantaneously without retraining delays."

## Slide 11: Results
- Final Predictions generated on a scale of 0 to 20 dynamically.
- Implemented category thresholds: Needs Improvement, Average, Good, Excellent.
- Automatically isolates Feature Importances (Internal Marks dominate context).
- **Verbal Notes**: "Our final system not only outputs a highly reliable score out of 20, but it bins this into readable categories giving the educator a rapid insight context along with structural recommendations."

## Slide 12: Future Scope & Conclusion
- Incorporate more behavioral real-world data (LMS logs, library access).
- Explore cross-subject analytics tracking.
- Test generalized performance utilizing deep neural architectures.
- Extracted models successfully automate early-warning detection.
- **Verbal Notes**: "To conclude, ML proves highly effective at academic prognosis. In the future, this system could be embedded into official Learning Management Systems taking real-time mouse-activity or library login data into account. Thank you."
