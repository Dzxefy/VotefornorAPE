# radiomics.py （改造后）
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np
import warnings

def single_patient_feature_extractor(image_path, mask_path, extractor_settings=None):
    """
    单患者影像组学特征提取，返回和训练时列顺序一致的单行DataFrame
    """
    # 默认提取参数，必须和你训练时完全一致！否则特征分布漂移模型会废
    if extractor_settings is None:
        extractor_settings = {
            'binWidth': 10,
            'sigma': [3, 4, 5],
            'resampledPixelSpacing': [1, 1, 1],
            'voxelArrayShift': 1000,
            'normalize': True,
            'normalizeScale': 100,
            'force2D': False,  # 不强制使用2D特征
            'interpolator': 'sitkNearestNeighbor',
            'multiprocessing': True,
            'removeOutliers': True
        }
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_settings)
    #extractor.enableAllImageTypes()   

    #先禁用全部图像类型
    extractor.disableAllImageTypes()
    for img_type in {'Wavelet','LoG','Square','Logarithm','Gradient','LocalBinaryPattern'}:  #指定要哪些图像类型
        extractor.enableImageTypeByName(img_type)

    #先禁用全部特征类型
    extractor.disableAllFeatures()
    for fc in {'firstorder','glcm','glrlm','glszm','gldm','ngtdm'}:
        extractor.enableFeatureClassByName(fc)   # ← 整类全开
    # 读图
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    
    # 你原来的mask过滤逻辑（去掉造影剂标签10）
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
    
    # 提特征
    feat_dict = extractor.execute(image, mask)
    
    # 【关键】只保留 radiomics 原生特征（去掉diagnostics_开头的和你自己加的ID/nongdu）
    # 如果你训练时确实用了nongdu，那把下面这行删掉，保留nongdu即可
    filtered_feat = {k: v for k, v in feat_dict.items() 
                    if not k.startswith('diagnostics_') and k != 'nongdu'}
    
    # 转单行DataFrame
    df = pd.DataFrame([filtered_feat])
    
    # 可选：如果你训练时保留了nongdu，这里单独算塞回去
    # image_array = sitk.GetArrayFromImage(image)
    # mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    # df['nongdu'] = np.mean(image_array[mask_array == 10]) if (mask_array==10).any() else 0
    
    return df


def _filter_mask(mask):
    """你原来的mask过滤逻辑，非0非10改成1（提征ROI），10（造影剂）排除"""
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array[(mask_array != 0) & (mask_array != 10)] = 1
    mask_array[mask_array == 10] = 0
    
    new_mask = sitk.GetImageFromArray(mask_array)
    new_mask.SetOrigin(mask.GetOrigin())
    new_mask.SetSpacing(mask.GetSpacing())
    new_mask.SetDirection(mask.GetDirection())
    return new_mask
