import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Import from the simplified class definition
from UnifiedRFESVCEnsemble import UnifiedRFESVCEnsemble

# 1. Cache loading of model and preprocessing objects
@st.cache_resource
def load_resources():
    # Use relative paths to ensure files are in correct location
    model_path = "saved_models/ensemble_model.pkl"
    features_path = "saved_models/feature_list.pkl"
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    if not os.path.exists(features_path):
        st.error(f"Feature file not found: {features_path}")
        return None, None
    
    try:
        model = joblib.load(model_path)
        expected_features = joblib.load(features_path)
        print(f"expected_features:\n{expected_features}")
        st.success("Model loaded successfully!")
        return model, expected_features
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

# 2. Page configuration
st.set_page_config(page_title="Pulmonary Embolism Prognosis Assessment", layout="wide")
st.title("Radiomics-Based Prognostic Assessment Tool for Normotensive Acute Pulmonary Embolism Patients")
st.write("Upload extracted feature Excel file and manually input RV/LV values for single-patient prediction")

# 3. Load resources
model, expected_features = load_resources()

if model is None or expected_features is None:
    st.stop()

# 4. Manual input for RV/LV values
st.subheader("Manual Input for RV/LV Values")
col1, col2 = st.columns(2)
with col1:
    rv_value = st.number_input("Enter RV diameter", min_value=0.0, step=0.1, value=0.0, 
                              help="Right ventricular diameter")
with col2:
    lv_value = st.number_input("Enter LV diameter", min_value=0.0, step=0.1, value=0.0,
                              help="Left ventricular diameter")

# 5. Upload feature table
st.subheader("Upload Radiomics Feature Table")
uploaded_file = st.file_uploader("Select pyradiomics-extracted feature file (.xlsx format)", 
                                 type=["xlsx"], 
                                 help="Please ensure the file contains data for only one patient")

if uploaded_file is not None:
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        
        # Check number of data rows
        if len(df) > 1:
            st.error("Only one patient's data can be uploaded at a time. Multiple rows detected.")
            st.stop()
        
        st.success("File read successfully!")
        st.subheader("Data Preview")
        st.dataframe(df)
        
        # Add RV/LV columns
        df["RV"] = rv_value
        df["LV"] = lv_value
        
        # Check feature completeness
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        if missing_features:
            st.error(f"Missing required feature columns: {missing_features}")
            st.info(f"Required feature columns: {expected_features}")
        else:
            # Extract features in the same order as training
            infer_df = df[expected_features]
            print(f"infer_df:\n{infer_df}")
            
            # Convert to numpy array (model requires array input)
            X_input = infer_df.values
            print(f"X_input:\n{X_input}")
            
            # Prediction button
            if st.button("Start Prediction", type="primary"):
                with st.spinner("Calculating prediction results..."):
                    try:
                        # Get prediction probability
                        pred_proba = model.predict(X_input)[0]
                        
                        # Apply threshold rule
                        threshold = 0.3571
                        pred_label = 1 if pred_proba > threshold else 0
                        
                        # Calculate RV/LV ratio
                        rv_lv_ratio = rv_value / lv_value if lv_value != 0 else float('inf')
                        
                        # Display results
                        st.subheader("📊 Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model Prediction Probability", f"{pred_proba:.4f}")
                        with col2:
                            st.metric("Predicted Label", f"{pred_label}")
                        with col3:
                            st.metric("RV/LV Ratio", f"{rv_lv_ratio:.4f}")
                        
                        # Threshold explanation
                        st.info(f"Threshold Rule: Probability > {threshold} → Label 1, otherwise Label 0")
                        
                        # Final assessment
                        st.subheader("🎯 Prognostic Assessment")
                        if pred_label == 1 and rv_lv_ratio > 1:
                            st.error("⚠️ **Prediction Result: Poor 30-day prognosis for this patient**")
                            st.write("Recommendation: Requires close monitoring and aggressive treatment")
                        else:
                            st.success("✅ **Prediction Result: Good 30-day prognosis for this patient**")
                            #st.write("Recommendation: Routine follow-up monitoring")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        
    except Exception as e:
        st.error(f"File processing failed: {str(e)}")
        st.write("Please check if the file format is correct")
