# app.py （最顶部加入）
import os
import sys

# 1. 确保 .streamlit 目录存在
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMLIT_DIR = os.path.join(BASE_DIR, ".streamlit")
CONFIG_PATH = os.path.join(STREAMLIT_DIR, "config.toml")

if not os.path.exists(STREAMLIT_DIR):
    os.makedirs(STREAMLIT_DIR)

# 2. 检查并写入配置
if not os.path.exists(CONFIG_PATH):
    print("Creating Streamlit config file for large file uploads...")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("[server]\n")
        f.write("maxUploadSize = 1024\n")  # 设置为 1024 MB
    print("Config created. Please restart the app if it was already running.")
else:
    # 可选：检查现有配置是否足够大
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        if "maxUploadSize" not in content:
            with open(CONFIG_PATH, "a", encoding="utf-8") as f:
                f.write("\n[server]\nmaxUploadSize = 1024\n")
        # 简单检查数值（如果需要更复杂的解析，建议用 configparser 或 toml 库）
        elif "maxUploadSize = 200" in content:
             print("Warning: maxUploadSize is set to 200MB. Consider increasing it for medical images.")


# app.py （精简最终版）
import streamlit as st
import pandas as pd
import joblib
import tempfile
import os
from radiomicspipeline import single_patient_feature_extractor
from UnifiedRFESVCEnsemble import UnifiedRFESVCEnsemble

# 1. 加载模型和特征列表（缓存）
@st.cache_resource
def load_resources():
    model = joblib.load("./saved_models/ensemble_model.pkl")
    expected_features = joblib.load("./saved_models/feature_list.pkl")
    return model, expected_features

model, expected_features = load_resources()

# 2. 页面 UI
st.title("Radiomics-Based Prognostic Assessment Tool for Normotensive Acute Pulmonary Embolism Patients")
st.write("Upload CT image + segmentation mask, auto-extract radiomics features and predict 30-day prognosis.")

# 3. RV/LV 手动输入 —— 改成 optional，默认 0 表示"未输入"
col1, col2 = st.columns(2)
with col1:
    rv_value = st.number_input("RV diameter (mm, optional)", min_value=0.0, step=0.1, value=0.0)
with col2:
    lv_value = st.number_input("LV diameter (mm, optional)", min_value=0.0, step=0.1, value=0.0)

# 简单判定：用户两个都没动（仍是 0）就算没输入
rvlv_provided = not (rv_value == 0.0 and lv_value == 0.0)

# 4. 图像 / mask 上传（唯一入口，xlsx 分支已删除）
st.subheader("Upload Image + Mask")
img_file = st.file_uploader(
    "CT image (.nii)",
    type=["nii"],
    key="img"
)
mask_file = st.file_uploader(
    "Segmentation mask (.nii)",
    type=["nii"],
    key="mask"
)

# 5. 预测主逻辑
if st.button("Start Prediction"):
    if img_file is None or mask_file is None:
        st.warning("Please upload both CT image and segmentation mask.")
        st.stop()

    try:
        # 5.1 存临时 nii（SimpleITK 只能读文件路径）
        with tempfile.NamedTemporaryFile(suffix=f".{img_file.name.split('.')[-1]}", delete=False) as tmp_img:
            tmp_img.write(img_file.getbuffer())
            img_path = tmp_img.name
        with tempfile.NamedTemporaryFile(suffix=f".{mask_file.name.split('.')[-1]}", delete=False) as tmp_mask:
            tmp_mask.write(mask_file.getbuffer())
            mask_path = tmp_mask.name

        # 5.2 自动提征
        with st.spinner("Extracting radiomics features..."):
            feat_df = single_patient_feature_extractor(img_path, mask_path)

        # 5.3 RV/LV —— 只有用户输入了才拼（⚠ 前提：训练时 feature_list.pkl 里本来就有 RV/LV，
        #     如果训练时没把它们当特征，这两行其实会被下一行的 expected_features 切片滤掉，
        #     不影响模型；但 ratio 展示还能用）
        if rvlv_provided:
            feat_df["RV"] = rv_value
            feat_df["LV"] = lv_value

        # 5.4 对齐训练特征列
        missing = [f for f in expected_features if f not in feat_df.columns]
        if missing:
            st.error(f"Extracted features missing columns (check radiomics settings): {missing[:5]}...")
            st.stop()

        infer_df = feat_df[expected_features]
        st.success("Feature extraction done!")
        st.dataframe(infer_df)

    finally:
        if 'img_path' in locals() and os.path.exists(img_path):
            os.remove(img_path)
        if 'mask_path' in locals() and os.path.exists(mask_path):
            os.remove(mask_path)

    # 6. 预测 + 展示
    pred_proba = model.predict(infer_df)[0]
    pred_label = 1 if pred_proba > 0.3571 else 0

    st.success("Prediction completed!")

    st.subheader("Model Prediction Result")
    st.write(f"Prediction probability: **{pred_proba:.4f}**")
    st.write(f"Predicted label (threshold 0.3571): **{pred_label}**")

    # 6.1 RV/LV 只在用户输入了才出
    if rvlv_provided:
        rv_lv_ratio = rv_value / lv_value if lv_value != 0 else float('inf')
        st.subheader("RV/LV Ratio")
        st.write(f"RV/LV ratio: **{rv_lv_ratio:.4f}**")

        st.subheader("Final Summary")
        if pred_label == 1 and rv_lv_ratio > 1:
            st.error("🔴 Predicted poor 30-day prognosis")
        else:
            st.success("🟢 Predicted good 30-day prognosis")
    else:
        st.info("RV/LV not provided — showing model output only.")
        st.subheader("Final Summary")
        if pred_label == 1:
            st.error("🔴 Model predicts poor 30-day prognosis (probability > 0.3571)")
        else:
            st.success("🟢 Model predicts good 30-day prognosis (probability ≤ 0.3571)") 
