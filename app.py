import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

st.set_page_config(page_title="Student Performance Prediction System", layout="wide", page_icon="🎓")

# --- LOAD MODELS & METADATA ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/best_model.pkl')
    metadata = joblib.load('models/model_metadata.pkl')
    return model, metadata

try:
    model, metadata = load_assets()
except Exception as e:
    st.error("Model artifacts not found. Please run `python src/train_model.py` first.")
    st.stop()

# --- CONFIGURABLE CONSTANTS ---
CAT_EXCELLENT = 16
CAT_GOOD = 13
CAT_AVERAGE = 10

st.title("🎓 Student Performance Prediction System")
st.markdown("A Machine Learning Mini-Project evaluating factors affecting final examination performance.")

# --- TABS ---
tab1, tab2 = st.tabs(["🎯 Predict Student Performance", "📊 Analytics & Model Dashboard"])

with tab1:
    st.header("Enter Student Information")
    st.markdown("Provide the student's details below to predict their final score (0 - 20).")
    
    with st.sidebar:
        st.header("📝 Student Inputs")
        st.markdown("*(Adjust core features here)*")
        
        G1 = st.slider("First Period Grade (G1)", 0, 20, 10, help="Internal Marks 1")
        G2 = st.slider("Second Period Grade (G2)", 0, 20, 10, help="Internal Marks 2")
        absences = st.slider("Number of Absences", 0, 93, 2, help="Lower absences mean better attendance.")
        studytime = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4], format_func=lambda x: {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}[x])
        failures = st.slider("Past Class Failures", 0, 3, 0)
        Medu = st.selectbox("Mother's Education", options=[0,1,2,3,4], format_func=lambda x: ["None", "Primary", "5th-9th grade", "Secondary", "Higher Education"][x], index=4)
        Fedu = st.selectbox("Father's Education", options=[0,1,2,3,4], format_func=lambda x: ["None", "Primary", "5th-9th grade", "Secondary", "Higher Education"][x], index=2)
        higher = st.radio("Wants Higher Education?", ["yes", "no"], index=0)
        internet = st.radio("Internet at Home?", ["yes", "no"], index=0)
        goout = st.slider("Going out with friends (1-5)", 1, 5, 3)

    # For the remaining 22 features, we just use typical/median values to construct the dataframe, reducing UI clutter but remaining mathematically correct.
    default_data = {
        'school': 'GP', 'sex': 'F', 'age': 16, 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',
        'Mjob': 'other', 'Fjob': 'other', 'reason': 'course', 'guardian': 'mother',
        'traveltime': 1, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no', 'activities': 'yes',
        'nursery': 'yes', 'romantic': 'no', 'famrel': 4, 'freetime': 3, 'Dalc': 1, 'Walc': 1, 'health': 3
    }
    
    input_dict = default_data.copy()
    input_dict.update({
        'G1': G1, 'G2': G2, 'absences': absences, 'studytime': studytime,
        'failures': failures, 'Medu': Medu, 'Fedu': Fedu, 'higher': higher,
        'internet': internet, 'goout': goout
    })
    
    df_input = pd.DataFrame([input_dict])
    
    col_pred1, col_pred2 = st.columns([1, 1])
    
    with col_pred1:
        st.markdown("### Ready to predict?")
        if st.button("Predict Performance", type="primary", use_container_width=True):
            pred = model.predict(df_input)[0]
            pred = np.clip(pred, 0, 20)
            
            st.success("Prediction complete!")
            
            # --- 11. PREDICTION OUTPUT ---
            st.markdown(f"### Predicted Final Exam Score: **{pred:.2f} / 20**")
            
            scaled_score = (pred / 20.0) * 100
            
            # --- 12. PERFORMANCE CATEGORY ---
            if pred >= CAT_EXCELLENT:
                category = "Excellent"
                color = "green"
            elif pred >= CAT_GOOD:
                category = "Good"
                color = "blue"
            elif pred >= CAT_AVERAGE:
                category = "Average"
                color = "orange"
            else:
                category = "Needs Improvement"
                color = "red"
                
            st.markdown(f"#### Category: :{color}[{category}]")
            st.caption("*Note: These categories are project-defined interpretations and not official academic grading rules.*")
            
            # --- 13. ACADEMIC SUPPORT RECOMMENDATION ---
            st.markdown("---")
            st.markdown("#### 💡 Recommendation")
            if category == "Needs Improvement":
                st.warning("Rule-based suggestion: Student may require additional academic support. Consider improving study consistency, attendance, and internal assessment performance.")
            elif category == "Average":
                st.info("Rule-based suggestion: Student is performing moderately well. Improving study hours and internal assessment preparation may help push the final score higher.")
            else:
                st.success("Rule-based suggestion: Student is showing good expected performance. Maintaining current academic habits is recommended.")
                
            st.markdown("---")
            with st.expander("View Full Input Vector Used"):
                st.dataframe(df_input)

with tab2:
    st.header("📊 Analytics & Model Dashboard")
    st.markdown("Data insights from the Kaggle Student Performance Dataset.")
    
    # --- 14. DASHBOARD ---
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Dataset Size", metadata['dataset_size'])
    m_col2.metric("Number of Features", metadata['num_features'])
    m_col3.metric("Best Model", metadata['best_model_name'])
    m_col4.metric("Test R² Score", f"{metadata['best_metrics']['r2_score']:.2f}")
    
    st.markdown("### 🏆 Model Comparison")
    results_df = pd.DataFrame(metadata['results_df'])
    st.dataframe(results_df, use_container_width=True)
    
    st.markdown("### 📈 Important Features (Random Forest)")
    if os.path.exists("reports/figures/feature_importance.png"):
        st.image("reports/figures/feature_importance.png", width=700)
    
    st.markdown("### 📊 Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("reports/figures/g1_vs_g3.png"):
            st.image("reports/figures/g1_vs_g3.png")
        if os.path.exists("reports/figures/studytime_vs_g3.png"):
            st.image("reports/figures/studytime_vs_g3.png")
    with c2:
        if os.path.exists("reports/figures/g3_distribution.png"):
            st.image("reports/figures/g3_distribution.png")
        if os.path.exists("reports/figures/absences_vs_g3.png"):
            st.image("reports/figures/absences_vs_g3.png")

