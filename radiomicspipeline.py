# radiomics.py（Streamlit 稳定版 · 不依赖特征名推断）
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np
import warnings

# ✅ 你指定的特征列表（完全不变）
TARGET_FEATURES = [
    'wavelet-HLH_glcm_ClusterShade',
    'wavelet-HLL_glszm_LargeAreaLowGrayLevelEmphasis',
    'lbp-3D-m1_glrlm_LongRunLowGrayLevelEmphasis',
    'lbp-3D-k_glcm_MaximumProbability',
    'log-sigma-4-mm-3D_firstorder_Median',
    'log-sigma-5-mm-3D_firstorder_10Percentile',
    'log-sigma-4-mm-3D_firstorder_Skewness',
    'log-sigma-3-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis',
    'lbp-3D-k_glrlm_LongRunLowGrayLevelEmphasis',
    'logarithm_glcm_MaximumProbability',
    'log-sigma-5-mm-3D_firstorder_Skewness',
    'lbp-3D-k_glcm_Imc1',
    'logarithm_gldm_DependenceNonUniformityNormalized',
    'lbp-3D-k_glszm_HighGrayLevelZoneEmphasis',
    'gradient_gldm_DependenceNonUniformityNormalized',
    'lbp-3D-m2_glcm_Correlation',
    'log-sigma-3-mm-3D_ngtdm_Busyness',
    'lbp-3D-m1_glcm_Correlation',
    'square_firstorder_90Percentile',
    'lbp-3D-m2_glrlm_ShortRunLowGrayLevelEmphasis',
    'logarithm_gldm_SmallDependenceEmphasis',
    'lbp-3D-m1_gldm_SmallDependenceLowGrayLevelEmphasis',
    'square_glcm_ClusterShade',
    'lbp-3D-m2_glszm_SizeZoneNonUniformityNormalized',
    'wavelet-LLL_glrlm_RunEntropy',
    'lbp-3D-m2_glszm_SmallAreaEmphasis',
    'log-sigma-5-mm-3D_gldm_SmallDependenceEmphasis',
    'lbp-3D-k_glrlm_ShortRunHighGrayLevelEmphasis'
]

def _get_enabled_image_types():
    """只返回你用到的 imageType（最小化）"""
    return {
        'wavelet',
        'log-sigma-3-mm-3D',
        'log-sigma-4-mm-3D',
        'log-sigma-5-mm-3D',
        'lbp-3D-m1',
        'lbp-3D-m2',
        'lbp-3D-k',
        'logarithm',
        'square',
        'gradient'
    }

def _get_enabled_feature_classes():
    """只返回你用到的 featureClass（不碰特征名）"""
    return {
        'firstorder',
        'glcm',
        'glrlm',
        'glszm',
        'gldm',
        'ngtdm'
    }

def single_patient_feature_extractor(image_path, mask_path, extractor_settings=None):
    """
    ✅ Streamlit 稳定版
    只启用必要的 imageType + featureClass
    提取后再按 TARGET_FEATURES 精确过滤
    """

    if extractor_settings is None:
        extractor_settings = {
            'binWidth': 10,
            'sigma': [3, 4, 5],
            'resampledPixelSpacing': [1, 1, 1],
            'voxelArrayShift': 1000,
            'normalize': True,
            'normalizeScale': 100,
            'force2D': False,
            'interpolator': sitk.sitkBSpline,
            'removeOutliers': 3.0,
        }

    extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_settings)

    # ✅ 1. 最小化 imageType（关键）
    extractor.disableAllImageTypes()
    for img_type in _get_enabled_image_types():
        extractor.enableImageTypeByName(img_type)

    # ✅ 2. 最小化 featureClass（不指定特征名！）
    extractor.disableAllFeatures()
    for fc in _get_enabled_feature_classes():
        extractor.enableFeatureClassByName(fc)

    # ✅ 3. 读图 & mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    mask = _filter_mask(mask)

    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    roi_vals = img_arr[mask_arr > 0]

    if roi_vals.size == 0:
        raise RuntimeError("❌ Mask 过滤后 ROI 为空")
    if roi_vals.std() == 0:
        raise RuntimeError("❌ ROI 内灰度值完全相同，无法离散化")

    # ✅ 4. 提取（现在只算 ~100 个特征，不是 1500+）
    feat_dict = extractor.execute(image, mask)

    # ✅ 5. 严格按你给的列表过滤（列顺序 100% 一致）
    row = {}
    missing = []
    for feat in TARGET_FEATURES:
        if feat in feat_dict:
            row[feat] = feat_dict[feat]
        else:
            row[feat] = np.nan
            missing.append(feat)

    if missing:
        warnings.warn(f"缺失特征（ROI 过小或参数不匹配）: {missing}")

    return pd.DataFrame([row], columns=TARGET_FEATURES)


def _filter_mask(mask):
    mask_arr = sitk.GetArrayFromImage(mask)
    mask_arr[(mask_arr != 0) & (mask_arr != 10)] = 1
    mask_arr[mask_arr == 10] = 0

    new_mask = sitk.GetImageFromArray(mask_arr)
    new_mask.SetOrigin(mask.GetOrigin())
    new_mask.SetSpacing(mask.GetSpacing())
    new_mask.SetDirection(mask.GetDirection())
    return new_mask
