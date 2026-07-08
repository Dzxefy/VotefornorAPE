import streamlit as st
import pandas as pd
import joblib
import tempfile
import os
import pathlib
from radiomicspipeline import single_patient_feature_extractor
from UnifiedRFESVCEnsemble import UnifiedRFESVCEnsemble

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

# 4. 图像 / mask 上传（支持 .nii 和 .nii.gz）
st.subheader("Upload Image + Mask")
img_file = st.file_uploader(
    "CT image",
    type=["nii", "nii.gz"],  # 支持 .nii 和 .nii.gz
    key="img"
)
mask_file = st.file_uploader(
    "Segmentation mask)",
    type=["nii", "nii.gz"],  # 支持 .nii 和 .nii.gz
    key="mask"
)

# 5. 预测主逻辑
if st.button("Start Prediction"):
    if img_file is None or mask_file is None:
        st.warning("Please upload both CT image and segmentation mask.")
        st.stop()

    # 获取正确的文件后缀（智能处理 .nii.gz）
    def get_correct_suffix(filename):
        """智能获取文件后缀，正确处理 .nii.gz"""
        p = pathlib.Path(filename)
        # 检查是否是 .nii.gz
        if p.name.endswith('.nii.gz'):
            return '.nii.gz'
        # 否则返回普通后缀
        return p.suffix
    
    img_suffix = get_correct_suffix(img_file.name)
    mask_suffix = get_correct_suffix(mask_file.name)
    
    # 临时文件路径
    img_path = None
    mask_path = None
    
    try:
        # 5.1 存临时文件到 /tmp 目录（SimpleITK 只能读文件路径）
        with tempfile.NamedTemporaryFile(
            suffix=img_suffix, 
            delete=False, 
            dir="/tmp"
        ) as tmp_img:
            tmp_img.write(img_file.getbuffer())
            img_path = tmp_img.name
            
        with tempfile.NamedTemporaryFile(
            suffix=mask_suffix, 
            delete=False, 
            dir="/tmp"
        ) as tmp_mask:
            tmp_mask.write(mask_file.getbuffer())
            mask_path = tmp_mask.name

        # 验证文件后缀是否正确
        st.info(f"Image saved as: {img_path}")
        st.info(f"Mask saved as: {mask_path}")
        
        # 5.2 自动提征
        with st.spinner("Extracting radiomics features..."):
            feat_df = single_patient_feature_extractor(img_path, mask_path)

        # 5.3 RV/LV —— 只有用户输入了才拼
        if rvlv_provided:
            feat_df["RV"] = rv_value
            feat_df["LV"] = lv_value

        # 5.4 对齐训练特征列
        missing = [f for f in expected_features if f not in feat_df.columns]
        if missing:
            st.error(f"Extracted features missing columns (check radiomics settings): {missing}")
            st.stop()

        infer_df = feat_df[expected_features]
        st.success("Feature extraction done!")
        st.dataframe(infer_df)

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        raise  # 重新抛出异常以便调试
        
    finally:
        # 5.5 清理临时文件（增强版）
        for path_var, path_name in [(img_path, "image"), (mask_path, "mask")]:
            if path_var and os.path.exists(path_var):
                try:
                    os.remove(path_var)
                    st.info(f"Cleaned up temporary {path_name} file")
                except Exception as e:
                    st.warning(f"Could not remove temporary {path_name} file: {e}")

    # 1. 加载模型和特征列表（缓存）- 使用绝对路径
    @st.cache_resource
    def load_resources():
        base = pathlib.Path(__file__).parent
        model = joblib.load(base / "saved_models" / "ensemble_model.pkl")
        expected_features = joblib.load(base / "saved_models" / "feature_list.pkl")
        return model, expected_features
    
    model, expected_features = load_resources()
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

