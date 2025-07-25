import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier # <--- NEW IMPORT ADDED HERE
from sklearn.linear_model import LogisticRegression # <--- ADDED THIS TOO for completeness if you ever switch

# --- URL for Your GitHub Image ---
GITHUB_HEART_IMAGE_URL = "https://raw.githubusercontent.com/Kennt96/capstone-stroke-predictor-app/main/Heart.jpg"

# --- Configuration for the Streamlit App ---
st.set_page_config(
    page_title="Heart Stroke Risk Predictor",
    page_icon=GITHUB_HEART_IMAGE_URL,
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load the Trained Model Pipeline ---
pipeline = None
try:
    pipeline = joblib.load('stroke_prediction_pipeline.pkl')
    st.success("Model pipeline loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'stroke_prediction_pipeline.pkl' not found.")
    st.info("Please ensure the model was saved correctly from your training script and is in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error("An unexpected error occurred while loading the model pipeline.")
    st.error(f"**Detailed Loading Error:** {e}")
    st.info("This often indicates a scikit-learn version mismatch between where the model was saved and where it's being loaded. Ensure your `requirements.txt` specifies the exact scikit-learn version used during model training (e.g., `scikit-learn==1.6.1`).")
    st.stop()

if pipeline is None:
    st.stop()

# --- Define Feature Lists (as used in training after VIF) ---
numerical_features_for_model = ['age', 'avg_glucose_level']
categorical_features_for_model = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                                  'work_type', 'Residence_type', 'smoking_status']


# --- App Title and Description ---
st.title("Heart Stroke Risk Predictor")
st.image(GITHUB_HEART_IMAGE_URL, width=120)

# Create tabs for navigation
tab1, tab2, tab3 = st.tabs(["Overview", "Most Important Features", "Recommendations"])

with tab1:
    st.markdown("### **Overview**")
    st.markdown("""
        This application predicts the likelihood of a patient experiencing a stroke
        based on various health and demographic attributes.
        Please enter the patient's details in the sidebar.
    """)

    st.write("---")

    # --- Sidebar for User Input ---
    st.sidebar.markdown("### **Patient Data Input**")

    # Define input widgets for each feature
    gender_options = ['Male', 'Female']
    gender = st.sidebar.selectbox("Gender", gender_options)

    age = st.sidebar.slider("Age (years)", min_value=0.0, max_value=82.0, value=45.0, step=0.1)

    hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    ever_married = st.sidebar.selectbox("Ever Married", ['Yes', 'No'])

    work_type_options = ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked']
    work_type = st.sidebar.selectbox("Work Type", work_type_options)

    residence_type_options = ['Urban', 'Rural']
    Residence_type = st.sidebar.selectbox("Residence Type", residence_type_options)

    avg_glucose_level = st.sidebar.number_input("Average Glucose Level", min_value=55.0, max_value=170.0, value=90.0, step=0.1)

    bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

    smoking_status_options = ['never smoked', 'Unknown', 'formerly smoked', 'smokes']
    smoking_status = st.sidebar.selectbox("Smoking Status", smoking_status_options)

    # --- Prediction Logic ---
    input_data_df = pd.DataFrame([[
        gender, age, hypertension, heart_disease, ever_married,
        work_type, Residence_type, avg_glucose_level, bmi, smoking_status
    ]], columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ])

    # Button to trigger the prediction
    if st.sidebar.button("Predict Stroke Risk"):
        st.markdown("### **Prediction Results:**")

        try:
            prediction_proba = pipeline.predict_proba(input_data_df)[:, 1][0]
            prediction_class = pipeline.predict(input_data_df)[0]

            st.metric(label="Probability of Stroke", value=f"{prediction_proba * 100:.2f}%")

            if prediction_class == 1:
                st.error("⚠️ **HIGH RISK OF STROKE**")
                st.write("Based on the provided information, the model indicates a high likelihood of stroke.")
            else:
                st.success("✅ **LOW RISK OF STROKE**")
                st.write("Based on the provided information, the model indicates a low likelihood of stroke.")

        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")

    st.write("---")
    st.markdown("Developed using Streamlit and Scikit-learn.")

with tab2:
    st.markdown("### **Most Important Features**")
    st.markdown("""
    This section highlights the features that the model found most influential in predicting stroke risk.
    Feature importance helps us understand which patient attributes contribute most to the model's decisions.
    """)

    classifier = pipeline.named_steps['classifier']

    if isinstance(classifier, RandomForestClassifier):
        preprocessor = pipeline.named_steps['preprocessor']
        # Access the OneHotEncoder from the 'cat' transformer within the preprocessor
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_features_for_model)

        # Combine numerical and one-hot encoded feature names
        all_feature_names = numerical_features_for_model + list(onehot_feature_names)

        # Get feature importances from the Random Forest Classifier
        importances = classifier.feature_importances_

        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        })

        # Sort by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        st.markdown("#### **Top 15 Feature Importances (Table)**")
        st.dataframe(feature_importance_df.head(15), use_container_width=True)

        st.markdown("#### **Top 10 Feature Importances (Chart)**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), ax=ax, palette='viridis')
        ax.set_title('Top 10 Feature Importances')
        ax.set_xlabel('Relative Importance (Higher = More Impactful)')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif isinstance(classifier, LogisticRegression):
        st.info("The best performing model identified during training was Random Forest. Its feature importances are displayed here. For Logistic Regression, feature importance is typically interpreted from its coefficients, which indicate the strength and direction of a feature's relationship with the log-odds of stroke.")
    else:
        st.warning("Feature importance display is currently configured for Random Forest Classifier, which was the best performing model. The type of classifier in the pipeline is not recognized for feature importance display.")


with tab3:
    st.markdown("### **Recommendations for Heart Stroke Prevention**")
    st.markdown("""
    Based on general medical guidelines and common risk factors for stroke, here are some recommendations to help reduce the risk of heart stroke:

    * **Manage Blood Pressure:** High blood pressure is a leading risk factor for stroke. Regularly monitor your blood pressure and work with your doctor to keep it within a healthy range through lifestyle changes (diet, exercise) and medication if needed.
    * **Control Diabetes:** Diabetes significantly increases stroke risk. Maintain healthy blood sugar levels through diet, regular physical activity, and prescribed medications.
    * **Maintain a Healthy Weight:** Obesity can contribute to high blood pressure, diabetes, and high cholesterol, all of which are stroke risk factors. Aim for a healthy Body Mass Index (BMI) through balanced nutrition and consistent physical activity.
    * **Adopt a Heart-Healthy Diet:** Focus on a diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit intake of saturated and trans fats, cholesterol, sodium, and added sugars. The Mediterranean diet is often recommended.
    * **Engage in Regular Physical Activity:** Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week. This can include brisk walking, jogging, swimming, or cycling.
    * **Quit Smoking:** Smoking damages blood vessels and significantly increases stroke risk. Quitting smoking is one of the most impactful steps you can take to reduce your risk.
    * **Limit Alcohol Consumption:** If you drink alcohol, do so in moderation. Excessive alcohol intake can raise blood pressure and contribute to other stroke risk factors.
    * **Manage Existing Heart Conditions:** Work closely with your doctor to manage any pre-existing heart conditions such as atrial fibrillation (irregular heartbeat), heart failure, or high cholesterol. These conditions can increase stroke risk.
    * **Regular Medical Check-ups:** Schedule routine check-ups with your healthcare provider to monitor your overall health, screen for risk factors, and receive personalized advice.
    * **Know Your Family History:** Be aware of your family's medical history, especially regarding stroke, heart disease, and diabetes, as genetics can play a role.

    ---
    **Important Disclaimer:** These are general health recommendations. This application and its recommendations are for informational purposes only and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for personalized medical advice, diagnosis, and treatment plans for any health concerns.
    """)
