import streamlit as st
import pandas as pd
import joblib

# Load the trained model and columns
model = joblib.load('heart_disease_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("❤️ Heart Disease Risk Prediction System")
st.write("Enter patient clinical parameters below to evaluate the risk of heart disease.")

# Input fields based on standard 13 features
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1.0, max_value=120.0, value=54.0)
    sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")][0], format_func=lambda x: x[1])[0] # Simplified mapping
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed; 3 = reversible)", [1, 2, 3])

# Prediction button
if st.button("Predict Risk"):
    # Create input dataframe matching training columns
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                              columns=model_columns)
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease Detected (Confidence: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Confidence: {prediction_proba[0][0]*100:.2f}%)")
