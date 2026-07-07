# radiomics.py （最小化特征提取版）
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np
import warnings

# 你指定的特征列表（直接硬编码，不反推）
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

def _parse_target_features():
    """
    直接解析目标特征列表，不做任何推断
    返回：(image_types_to_enable, feature_config)
    """
    # 手动定义需要启用的图像类型（直接从特征名观察得出）
    image_types_to_enable = {
        'wavelet',           # wavelet-HLH, wavelet-HLL, wavelet-LLL
        'log-sigma-3-mm-3D', # log-sigma-3-mm-3D
        'log-sigma-4-mm-3D', # log-sigma-4-mm-3D
        'log-sigma-5-mm-3D', # log-sigma-5-mm-3D
        'lbp-3D-m1',         # lbp-3D-m1
        'lbp-3D-m2',         # lbp-3D-m2
        'lbp-3D-k',          # lbp-3D-k
        'logarithm',         # logarithm
        'square',            # square
        'gradient'           # gradient
    }
    
    # 手动定义特征类配置（直接从特征名观察得出）
    feature_config = {
        'firstorder': ['Median', '10Percentile', 'Skewness', '90Percentile'],
        'glcm': ['ClusterShade', 'MaximumProbability', 'Imc1', 'Correlation'],
        'glrlm': ['LongRunLowGrayLevelEmphasis', 'ShortRunLowGrayLevelEmphasis', 
                  'RunEntropy', 'ShortRunHighGrayLevelEmphasis'],
        'glszm': ['LargeAreaLowGrayLevelEmphasis', 'HighGrayLevelZoneEmphasis',
                  'SizeZoneNonUniformityNormalized', 'SmallAreaEmphasis'],
        'gldm': ['SmallDependenceHighGrayLevelEmphasis', 'DependenceNonUniformityNormalized',
                 'SmallDependenceEmphasis', 'SmallDependenceLowGrayLevelEmphasis',
                 'SmallDependenceEmphasis'],
        'ngtdm': ['Busyness']
    }
    
    return image_types_to_enable, feature_config

def single_patient_feature_extractor(image_path, mask_path, extractor_settings=None):
    """
    单患者影像组学特征提取，只提取TARGET_FEATURES中指定的特征
    """
    # 解析目标特征配置
    image_types_to_enable, feature_config = _parse_target_features()
    
    # 默认提取参数（已根据你的特征列表优化）
    if extractor_settings is None:
        extractor_settings = {
            'binWidth': 10,
            'sigma': [3, 4, 5],  # 只启用你用到的sigma值
            'resampledPixelSpacing': [1, 1, 1],
            'voxelArrayShift': 1000,
            'normalize': True,
            'normalizeScale': 100,
            'force2D': False,
            'interpolator': 'sitkBSpline',  # 图像用BSpline，mask内部强制NearestNeighbor
            # 移除 multiprocessing（单病人反而慢）
            'removeOutliers': 3.0,  # 使用数值而非布尔值
        }
    
    # 创建提取器
    extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_settings)
    
    # ===== 关键：最小化启用图像类型 =====
    extractor.disableAllImageTypes()
    for img_type in image_types_to_enable:
        extractor.enableImageTypeByName(img_type)
    
    # ===== 关键：最小化启用特征类 =====
    extractor.disableAllFeatures()
    for feature_class, features in feature_config.items():
        extractor.enableFeaturesByName(feature_class, features)
    
    # 读图
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    
    # mask过滤逻辑（去掉造影剂标签10）
    mask = _filter_mask(mask)

    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)

    roi_vals = img_arr[mask_arr > 0]

    print("ROI 体素数:", roi_vals.size)
    print("ROI 最小值:", roi_vals.min())
    print("ROI 最大值:", roi_vals.max())
    print("ROI std:", roi_vals.std())

    if roi_vals.size == 0:
        raise RuntimeError("❌ Mask 过滤后 ROI 为空")

    if roi_vals.std() == 0:
        raise RuntimeError("❌ ROI 内灰度值完全相同，无法离散化")
    
    # 提特征（现在只会提取我们启用的那些）
    feat_dict = extractor.execute(image, mask)
    
    # ===== 关键：只保留我们指定的特征 =====
    filtered_feat = {}
    missing_features = []
    
    for target_feat in TARGET_FEATURES:
        if target_feat in feat_dict:
            filtered_feat[target_feat] = feat_dict[target_feat]
        else:
            missing_features.append(target_feat)
            filtered_feat[target_feat] = np.nan  # 用NaN填充缺失特征
    
    if missing_features:
        warnings.warn(f"以下特征未提取到（可能ROI太小或参数不匹配）: {missing_features}")
    
    # 转单行DataFrame，确保列顺序与TARGET_FEATURES完全一致
    df = pd.DataFrame([filtered_feat], columns=TARGET_FEATURES)
    
    return df

def _filter_mask(mask):
    """mask过滤逻辑，非0非10改成1（提征ROI），10（造影剂）排除"""
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array[(mask_array != 0) & (mask_array != 10)] = 1
    mask_array[mask_array == 10] = 0
    
    new_mask = sitk.GetImageFromArray(mask_array)
    new_mask.SetOrigin(mask.GetOrigin())
    new_mask.SetSpacing(mask.GetSpacing())
    new_mask.SetDirection(mask.GetDirection())
    return new_mask

# 可选：批量提取时的优化版本（复用extractor）
class RadiomicsExtractorPool:
    """用于批量提取的提取器池，避免重复初始化"""
    def __init__(self, extractor_settings=None):
        image_types_to_enable, feature_config = _parse_target_features()
        
        if extractor_settings is None:
            extractor_settings = {
                'binWidth': 10,
                'sigma': [1, 2, 3, 4, 5],
                'resampledPixelSpacing': [1, 1, 1],
                'voxelArrayShift': 1000,
                'normalize': True,
                'normalizeScale': 100,
                'force2D': False,  # 不强制使用2D特征
                'interpolator': 'sitkNearestNeighbor',
                'multiprocessing': False,
                'removeOutliers': True
            }
        
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_settings)
        
        # 最小化配置
        self.extractor.disableAllImageTypes()
        for img_type in image_types_to_enable:
            self.extractor.enableImageTypeByName(img_type)
        
        self.extractor.disableAllFeatures()
        for feature_class, features in feature_config.items():
            self.extractor.enableFeaturesByName(feature_class, features)
    
    def extract(self, image_path, mask_path):
        """单次提取"""
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        mask = _filter_mask(mask)
        
        feat_dict = self.extractor.execute(image, mask)
        
        # 按固定顺序返回
        return [feat_dict.get(feat, np.nan) for feat in TARGET_FEATURES]
