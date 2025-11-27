import streamlit as st
import joblib
import numpy as np

# --- Load saved model ---

model = joblib.load("breast_cancer_svm_model.joblib")

# Streamlit UI

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("Breast Cancer Prediction App")
st.write("Select Yes or No for the following symptoms:")

# Symptoms input

symptoms_list = [
"Lump in the breast or underarm",
"Breast pain or tenderness",
"Nipple changes (inversion/discharge)",
"Skin changes (dimpling/redness/thickening)",
"Swelling in part of the breast"
]

symptoms = []
for symptom in symptoms_list:
 user_input = st.selectbox(symptom, ("No", "Yes"))
symptoms.append(1 if user_input == "Yes" else 0)

# Predict button

if st.button("Predict"):
 if len(symptoms) != 5:
  st.error("Please select all symptoms!")
else:
# Convert symptoms to feature array
 X_input_full = np.zeros((1, 30))
for i in range(30):
 X_input_full[0, i] = symptoms[i % len(symptoms)] * (0.8 + 0.1 * (i % 3))  # safer indexing


    # Make prediction
pred = model.predict(X_input_full)[0]
proba = model.predict_proba(X_input_full)[0][1]

    # Display result
if pred == 1:
        st.markdown("<h2 style='color:red'>Prediction: Malignant</h2>", unsafe_allow_html=True)
else:
        st.markdown("<h2 style='color:green'>Prediction: Benign</h2>", unsafe_allow_html=True)

    # Probability bar and details
st.progress(proba)
st.write(f"Probability of Malignant: {proba:.2f}")
st.info("Note: This is a machine learning prediction. Please consult a doctor for medical advice.")