import numpy as np
from sklearn.preprocessing import StandardScaler

class UnifiedRFESVCEnsemble:
    """精简版集成模型，仅用于预测"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.selected_model_indices = []
        self.n_selected_models = 0
        self.voting_threshold = 0.5
        self.models = []  # 每个元素: {'model': svc, 'selected_features': list}
    
    def predict(self, X):
        """预测新数据"""
        if not self.models:
            raise ValueError("No models loaded")
        
        if self.n_selected_models == 0:
            raise ValueError("No selected models available")
        
        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]
        n_features = X_scaled.shape[1]
        
        # ✅ 只使用选中的模型
        model_binary_predictions = np.zeros((n_samples, self.n_selected_models))
        
        for j, idx in enumerate(self.selected_model_indices):
            model_info = self.models[idx]
            model = model_info['model']
            selected_features = model_info['selected_features']
            
            # ✅ 防御性编程：过滤非法索引
            valid_features = [f for f in selected_features if f < n_features]
            
            if len(valid_features) == 0:
                # 如果没有有效特征，给出中性预测
                model_binary_predictions[:, j] = 0.5
            else:
                X_selected = X_scaled[:, valid_features]
                y_pred_proba = model.predict_proba(X_selected)[:, 1]
                y_pred_binary = (y_pred_proba > self.voting_threshold).astype(int)
                model_binary_predictions[:, j] = y_pred_binary
        
        vote_counts = np.sum(model_binary_predictions, axis=1)
        ensemble_pred = vote_counts / self.n_selected_models
        
        return ensemble_pred