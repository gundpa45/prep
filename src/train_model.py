import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_eda_plots(df, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribution of Final Exam Scores
    plt.figure(figsize=(8, 5))
    sns.histplot(df['G3'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Final Exam Scores (G3)')
    plt.xlabel('Final Score')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/g3_distribution.png')
    plt.close()

    # 2. Absences (Attendance Proxy) vs Final Score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='absences', y='G3', alpha=0.6, color='red')
    plt.title('Absences vs Final Exam Score')
    plt.xlabel('Number of Absences')
    plt.ylabel('Final Score')
    plt.savefig(f'{output_dir}/absences_vs_g3.png')
    plt.close()
    
    # 3. Study Hours (studytime) vs Final Score
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='studytime', y='G3', palette='viridis')
    plt.title('Study Time vs Final Exam Score')
    plt.xlabel('Study Time Category (1 = <2 hours, 4 = >10 hours)')
    plt.ylabel('Final Score')
    plt.savefig(f'{output_dir}/studytime_vs_g3.png')
    plt.close()

    # 4. Internal Marks (G1) vs Final Score (G3)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='G1', y='G3', alpha=0.6, color='green')
    plt.title('First Period Internal Marks (G1) vs Final Score (G3)')
    plt.xlabel('G1 Score')
    plt.ylabel('Final Score (G3)')
    plt.savefig(f'{output_dir}/g1_vs_g3.png')
    plt.close()

    # 5. Correlation Heatmap (Numeric)
    plt.figure(figsize=(10, 8))
    num_df = df.select_dtypes(include=[np.number])
    sns.heatmap(num_df.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Numerical Features Correlation Heatmap')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    
    print("EDA plots saved in 'reports/figures/'.")

def build_model_pipeline(num_cols, cat_cols, regressor):
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])
    
    return pipeline

def main():
    print("Loading data...")
    # Using ; as sep per inspection
    df = pd.read_csv('data/student-mat.csv', sep=';')
    
    print("Generating EDA...")
    create_eda_plots(df)
    
    # Feature Definition
    target = 'G3'
    X = df.drop(columns=[target])
    y = df[target]
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Train-test split (80% train, 20% test avoiding data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=5),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    best_r2 = -float("inf")
    best_model_name = ""
    best_pipeline = None
    
    print("Training and evaluating models...")
    for name, model in models.items():
        pipeline = build_model_pipeline(num_cols, cat_cols, model)
        pipeline.fit(X_train, y_train)
        
        # Test predictions
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Train predictions (To show to professor)
        y_train_pred = pipeline.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        
        results.append({
            'Model': name,
            'Test_MAE': f"{mae:.2f}",
            'Test_RMSE': f"{rmse:.2f}",
            'Test_R2': f"{r2:.2f}",
            'Train_R2': f"{train_r2:.2f}"
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    print(f"\nBest Model selected: {best_model_name} with Test R2: {best_r2:.2f}")
    
    # Getting Feature Importance for Random Forest / Decision Tree / Gradient Boosting
    # Pipeline contains 'preprocessor' and 'model'
    try:
        model_obj = best_pipeline.named_steps['model']
        preprocessor = best_pipeline.named_steps['preprocessor']
        
        if hasattr(model_obj, 'feature_importances_'):
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols)
            all_features = num_cols + list(encoded_cat_cols)
            
            importances = model_obj.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': all_features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(10)
            
            plt.figure(figsize=(10,6))
            sns.barplot(data=feature_imp_df, x='Importance', y='Feature', palette='magma')
            plt.title(f'Top 10 Feature Importances ({best_model_name})')
            plt.tight_layout()
            plt.savefig('reports/figures/feature_importance.png')
            plt.close()
            print("Saved Feature Importances chart.")
    except Exception as e:
        print("Could not extract feature importances natively.", str(e))

    # Saving Model and Meta Data
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, 'models/best_model.pkl')
    
    meta_data = {
        'dataset_size': len(df),
        'num_features': len(X.columns),
        'best_model_name': best_model_name,
        'best_metrics': {
            'Test_MAE': best_pipeline.predict(X_test),
            'r2_score': best_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, best_pipeline.predict(X_test)))
        },
        'results_df': results
    }
    joblib.dump(meta_data, 'models/model_metadata.pkl')
    
    print("\nTraining workflow completed successfully! Artifacts saved.")

if __name__ == "__main__":
    main()
