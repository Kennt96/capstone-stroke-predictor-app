import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64 # Import base64 for encoding SVG

# --- Futuristic Heart SVG as Data URI ---
# This SVG is embedded directly into the code.
# It's a stylized heart with some lines and a small circle to give a techy/futuristic look.
svg_content = """
<svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 21.35L10.55 20.03C5.4 15.36 2 12.27 2 8.5C2 5.42 4.42 3 7.5 3C9.24 3 10.91 3.81 12 5.08C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.42 22 8.5C22 12.27 18.6 15.36 13.45 20.03L12 21.35Z" fill="#ff4b4b"/>
  <path d="M12 21.35L10.55 20.03C5.4 15.36 2 12.27 2 8.5C2 5.42 4.42 3 7.5 3C9.24 3 10.91 3.81 12 5.08C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.42 22 8.5C22 12.27 18.6 15.36 13.45 20.03L12 21.35Z" stroke="#00ffff" stroke-width="0.5"/>
  <circle cx="12" cy="8.5" r="1" fill="#00ffff"/>
  <path d="M10 10L8 12L10 14" stroke="#00ffff" stroke-width="0.5"/>
  <path d="M14 10L16 12L14 14" stroke="#00ffff" stroke-width="0.5"/>
</svg>
"""
# Encode the SVG content to base64 for use as a data URI
FUTURISTIC_HEART_DATA_URI = f"data:image/svg+xml;base64,{base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')}"

# --- Configuration for the Streamlit App ---
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon=FUTURISTIC_HEART_DATA_URI, # Use the futuristic heart SVG for the browser tab icon
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("Stroke Risk Predictor") # Title text

# Display the futuristic heart image right below the title
# You can adjust the 'width' parameter to change its size on the page
st.image(FUTURISTIC_HEART_DATA_URI, width=120)

st.markdown("""
    This application predicts the likelihood of a patient experiencing a stroke
    based on various health and demographic attributes.
    Please enter the patient's details in the sidebar.
""")

st.write("---")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Data Input")

# Define input widgets for each feature
# Ensure these match the features and their expected types/categories used during training

# Gender
gender_options = ['Male', 'Female']
gender = st.sidebar.selectbox("Gender", gender_options)

# Age (using slider for range, based on original data's min/max/median)
# Original data age range: 0.08 to 82.0
age = st.sidebar.slider("Age (years)", min_value=0.0, max_value=82.0, value=45.0, step=0.1)

# Hypertension (binary)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Heart Disease (binary)
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Ever Married (binary)
ever_married = st.sidebar.selectbox("Ever Married", ['Yes', 'No'])

# Work Type (categorical)
work_type_options = ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked']
work_type = st.sidebar.selectbox("Work Type", work_type_options)

# Residence Type (binary)
residence_type_options = ['Urban', 'Rural']
Residence_type = st.sidebar.selectbox("Residence Type", residence_type_options)

# Average Glucose Level (using number_input for precision, based on original data's range)
# Original data avg_glucose_level range: 55.12 to 169.365 (after outlier treatment)
avg_glucose_level = st.sidebar.number_input("Average Glucose Level", min_value=55.0, max_value=170.0, value=90.0, step=0.1)

# BMI (using number_input for precision, based on original data's range)
# Original data bmi range: 10.3 to 46.3 (after outlier treatment)
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

# Smoking Status (categorical)
smoking_status_options = ['never smoked', 'Unknown', 'formerly smoked', 'smokes']
smoking_status = st.sidebar.selectbox("Smoking Status", smoking_status_options)

# --- Prediction Logic ---
# Create a DataFrame from the user inputs
# Ensure the column order matches the order of features expected by the trained pipeline
# (which is the order of columns in X_train from the original script)
input_data_df = pd.DataFrame([[
    gender, age, hypertension, heart_disease, ever_married,
    work_type, Residence_type, avg_glucose_level, bmi, smoking_status
]], columns=[
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
])

# Button to trigger prediction
if st.sidebar.button("Predict Stroke Risk"):
    st.write("### Prediction Results:")
    try:
        # Get probability of stroke (class 1)
        prediction_proba = pipeline.predict_proba(input_data_df)[:, 1][0]
        # Get the predicted class (0 or 1)
        prediction_class = pipeline.predict(input_data_df)[0]

        st.metric(label="Probability of Stroke", value=f"{prediction_proba * 100:.2f}%")

        if prediction_class == 1:
            st.error("⚠️ **HIGH RISK OF STROKE**")
            st.write("Based on the provided information, the model indicates a high likelihood of stroke.")
        else:
            st.success("✅ **LOW RISK OF STROKE**")
            st.write("Based on the provided information, the model indicates a low likelihood of stroke.")

        st.info("""
            **Disclaimer:** This prediction is generated by a machine learning model for informational purposes only.
            It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment.
            Always consult with a qualified healthcare provider for any health concerns.
        """)
    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")

st.write("---")
st.markdown("Developed using Streamlit and Scikit-learn.")
