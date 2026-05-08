import streamlit as st
import pandas as pd
import joblib
from UnifiedRFESVCEnsemble import UnifiedRFESVCEnsemble

# 1. Cache loading of model and preprocessing objects to avoid reloading on each interaction
@st.cache_resource
def load_resources():
    model = joblib.load("ensemble_model.pkl")  # Replace with your model path
    expected_features = joblib.load("feature_list.pkl")  # Feature names used during training
    return model, expected_features

model, expected_features = load_resources()

# 2. Page UI configuration
st.title("Radiomics-Based Prognostic Assessment Tool for Normotensive Acute Pulmonary Embolism Patients")
st.write("Upload your extracted feature Excel file and manually input RV/LV values for single-patient prediction")

# 3. Manual input for RV/LV values (supports decimals, with reasonable range)
col1, col2 = st.columns(2)
with col1:
    rv_value = st.number_input("Enter RV diameter", min_value=0.0, step=0.1, value=0.0)
with col2:
    lv_value = st.number_input("Enter LV diameter", min_value=0.0, step=0.1, value=0.0)

# 4. Upload xlsx feature table
uploaded_file = st.file_uploader("Upload radiomics features table from pyradiomics (.xlsx format)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read xlsx as DataFrame
        df = pd.read_excel(uploaded_file)
        
        # Check if only one patient (single row)
        if len(df) > 1:
            st.error("Please upload data for only ONE patient at a time. Multiple rows detected.")
            st.stop()
        
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        # Add RV/LV columns to DataFrame
        df["RV"] = rv_value
        df["LV"] = lv_value

        # Data validation: check if all required features from training are present
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        if missing_features:
            st.error(f"Uploaded feature table is missing required columns: {missing_features}. Please check your file.")
        else:
            # Extract features in the same order as training
            infer_df = df[expected_features]
            
            # Prediction button
            if st.button("Start Prediction"):
                # Get prediction probability
                pred_proba = model.predict(infer_df)[0]  # Single patient prediction
                
                # Apply threshold rule: >0.3571 → label 1, else label 0
                pred_label = 1 if pred_proba > 0.3571 else 0
                
                # Calculate RV/LV ratio
                rv_lv_ratio = rv_value / lv_value if lv_value != 0 else float('inf')
                
                # Display results
                st.success("Prediction completed!")
                
                # Print model prediction result
                st.subheader("Model Prediction Result")
                st.write(f"Model prediction probability: **{pred_proba:.4f}**")
                st.write(f"Predicted label (threshold 0.3571): **{pred_label}**")
                
                # Print RV/LV ratio
                st.subheader("RV/LV Ratio")
                st.write(f"RV/LV ratio: **{rv_lv_ratio:.4f}**")
                
                # Final summary
                st.subheader("Final Summary")
                if pred_label == 1 and rv_lv_ratio > 1:
                    st.error("Predicted poor 30-day prognosis for this patient")
                else:
                    st.success("Predicted good 30-day prognosis for this patient")
    except Exception as e:
        st.error(f"File processing failed: {str(e)}")