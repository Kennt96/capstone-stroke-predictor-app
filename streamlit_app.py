import streamlit as st
import pandas as pd
import joblib
import numpy as np
# base64 is no longer strictly needed if using raw GitHub image URLs,
# but keeping it doesn't hurt and might be useful for future changes.
import base64

# --- URL for Your GitHub Image ---
# This is the raw URL for your Heart.jpg on GitHub.
# Make sure 'Heart.jpg' is pushed to the root of your 'main' branch on GitHub.
GITHUB_HEART_IMAGE_URL = "https://raw.githubusercontent.com/Kennt96/capstone-stroke-predictor-app/main/Heart.jpg"

# --- Configuration for the Streamlit App ---
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon=GITHUB_HEART_IMAGE_URL, # This uses the URL for the browser tab icon
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load the Trained Model Pipeline ---
# This crucial block loads your saved machine learning model.
# It uses a try-except block for robust error handling during loading.
pipeline = None # Initialize pipeline to None before attempting to load
try:
    # Attempt to load the model pipeline from the .pkl file.
    # The file 'stroke_prediction_pipeline.pkl' must be in the same directory
    # as this 'streamlit_app.py' script on the deployed server.
    pipeline = joblib.load('stroke_prediction_pipeline.pkl')
    st.success("Model pipeline loaded successfully!") # Display success message if loading is successful
except FileNotFoundError:
    # Handle the case where the model file is not found.
    st.error("Error: Model file 'stroke_prediction_pipeline.pkl' not found.")
    st.info("Please ensure the model was saved correctly from your training script and is in the same directory as this app.")
    st.stop() # Stop the app execution if the model file is critical and missing
except Exception as e:
    # Catch any other general exceptions that might occur during model loading (e.g., version mismatch).
    st.error("An unexpected error occurred while loading the model pipeline.")
    st.error(f"**Detailed Loading Error:** {e}") # Display the exact error for debugging
    st.info("This often indicates a scikit-learn version mismatch between where the model was saved and where it's being loaded. Ensure your `requirements.txt` specifies the exact scikit-learn version used during model training (e.g., `scikit-learn==1.6.1`).")
    st.stop() # Stop the app execution if model loading fails

# If 'pipeline' is still None after the try-except block, it means loading failed.
# This check ensures that the rest of the app doesn't attempt to use a non-existent pipeline.
if pipeline is None:
    st.stop()


# --- App Title and Description ---
st.title("Stroke Risk Predictor") # Main title for the application

# Display the heart image from your GitHub repository right below the title.
# The 'width' parameter controls the size of the image on the page.
st.image(GITHUB_HEART_IMAGE_URL, width=120)

st.markdown("""
    This application predicts the likelihood of a patient experiencing a stroke
    based on various health and demographic attributes.
    Please enter the patient's details in the sidebar.
""")

st.write("---") # Adds a horizontal rule for visual separation

# --- Sidebar for User Input ---
st.sidebar.header("Patient Data Input") # Header for the input section in the sidebar

# Define input widgets for each feature.
# Ensure that the options and types for these widgets match the data
# that your trained model expects.

# Gender input: Selectbox with Male/Female options
gender_options = ['Male', 'Female']
gender = st.sidebar.selectbox("Gender", gender_options)

# Age input: Slider for a continuous range of age
# Min, max, and default values are set based on typical age ranges in stroke datasets.
age = st.sidebar.slider("Age (years)", min_value=0.0, max_value=82.0, value=45.0, step=0.1)

# Hypertension input: Selectbox for binary (0 or 1) choice, displayed as Yes/No
hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Heart Disease input: Selectbox for binary (0 or 1) choice, displayed as Yes/No
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Ever Married input: Selectbox for Yes/No
ever_married = st.sidebar.selectbox("Ever Married", ['Yes', 'No'])

# Work Type input: Selectbox for various work categories
work_type_options = ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked']
work_type = st.sidebar.selectbox("Work Type", work_type_options)

# Residence Type input: Selectbox for Urban/Rural
residence_type_options = ['Urban', 'Rural']
Residence_type = st.sidebar.selectbox("Residence Type", residence_type_options)

# Average Glucose Level input: Number input for a continuous numerical value
avg_glucose_level = st.sidebar.number_input("Average Glucose Level", min_value=55.0, max_value=170.0, value=90.0, step=0.1)

# BMI input: Number input for Body Mass Index
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

# Smoking Status input: Selectbox for different smoking categories
smoking_status_options = ['never smoked', 'Unknown', 'formerly smoked', 'smokes']
smoking_status = st.sidebar.selectbox("Smoking Status", smoking_status_options)

# --- Prediction Logic ---
# This section prepares the user's input data and makes a prediction using the loaded model.

# Create a Pandas DataFrame from the user inputs.
# The column names and their order MUST match the features and their order
# that your trained 'pipeline' expects.
input_data_df = pd.DataFrame([[
    gender, age, hypertension, heart_disease, ever_married,
    work_type, Residence_type, avg_glucose_level, bmi, smoking_status
]], columns=[
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
])

# Button to trigger the prediction
if st.sidebar.button("Predict Stroke Risk"):
    st.write("### Prediction Results:") # Subheader for results

    try:
        # Get the probability of the patient having a stroke (class 1).
        # pipeline.predict_proba returns probabilities for both classes (0 and 1).
        # We take the probability of class 1 (stroke).
        prediction_proba = pipeline.predict_proba(input_data_df)[:, 1][0]
        # Get the predicted class (0 for no stroke, 1 for stroke).
        prediction_class = pipeline.predict(input_data_df)[0]

        # Display the predicted probability as a metric.
        st.metric(label="Probability of Stroke", value=f"{prediction_proba * 100:.2f}%")

        # Display a message based on the predicted class.
        if prediction_class == 1:
            st.error("⚠️ **HIGH RISK OF STROKE**")
            st.write("Based on the provided information, the model indicates a high likelihood of stroke.")
        else:
            st.success("✅ **LOW RISK OF STROKE**")
            st.write("Based on the provided information, the model indicates a low likelihood of stroke.")

        # Display a disclaimer for medical advice.
        st.info("""
            **Disclaimer:** This prediction is generated by a machine learning model for informational purposes only.
            It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment.
            Always consult with a qualified healthcare provider for any health concerns.
        """)
    except Exception as e:
        # Catch any errors during the prediction process (e.g., incorrect input format).
        st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")

st.write("---") # Another horizontal rule
st.markdown("Developed using Streamlit and Scikit-learn.") # Footer
