import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class UnifiedRFESVCEnsemble:
    """Simplified RFE SVC Ensemble Model with Hard Voting"""
    
    def __init__(self, n_models=500, bootstrap_ratio=0.9, 
                 svc_params=None, min_features=1, max_features=None, 
                 rfe_step=1, random_state=42, 
                 voting_threshold=0.5, auc_threshold=0.7):
        self.n_models = n_models
        self.bootstrap_ratio = bootstrap_ratio
        self.min_features = min_features
        self.max_features = max_features
        self.rfe_step = rfe_step
        self.random_state = random_state
        self.voting_threshold = voting_threshold
        self.auc_threshold = auc_threshold
        np.random.seed(random_state)
        
        if svc_params is None:
            self.svc_params = {
                'kernel': 'linear',
                'probability': True,
                'C': 1.0,
                'random_state': random_state
            }
        else:
            self.svc_params = svc_params
            
        self.models = []
        self.feature_subsets = []
        self.feature_counts = []
        self.rfe_auc_curves = []
        self.model_performances = []
        self.scaler = StandardScaler()
        self.selected_model_indices = []
        self.n_selected_models = 0
        
    def balanced_bootstrap_sampling(self, X, y, n_samples_per_class=None):
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        
        if n_samples_per_class is None:
            n_pos = int(len(pos_indices) * self.bootstrap_ratio)
            n_neg = n_pos
        else:
            n_pos = n_samples_per_class
            n_neg = n_samples_per_class
        
        n_pos = min(n_pos, len(pos_indices))
        n_neg = min(n_neg, len(neg_indices))
        
        sampled_pos = np.random.choice(pos_indices, size=n_pos, replace=True)
        sampled_neg = np.random.choice(neg_indices, size=n_neg, replace=True)
        
        sampled_indices = np.concatenate([sampled_pos, sampled_neg])
        
        return X[sampled_indices], y[sampled_indices]
    
    def adaptive_rfe_for_single_model(self, X_train, y_train, X_val, y_val, feature_names=None):
        n_features = X_train.shape[1]
        
        if self.max_features is None:
            max_features_to_select = n_features
        else:
            max_features_to_select = min(self.max_features, n_features)
        
        X_train_scaled = X_train
        X_val_scaled = X_val
        
        auc_scores = []
        feature_subsets = []
        models = []
        
        current_features = list(range(n_features))
        best_auc = 0
        best_features = None
        best_model = None
        
        for n_select in range(max_features_to_select, self.min_features - 1, -self.rfe_step):
            if n_select <= 0:
                break
            
            if len(current_features) > n_select:
                svc_temp = SVC(**self.svc_params)
                svc_temp.fit(X_train_scaled[:, current_features], y_train)
                
                coefficients = svc_temp.coef_[0]
                feature_importance = np.abs(coefficients)
                least_important_idx = np.argmin(feature_importance)
                current_features.pop(least_important_idx)
            
            X_train_subset = X_train_scaled[:, current_features]
            X_val_subset = X_val_scaled[:, current_features]
            
            svc = SVC(**self.svc_params)
            svc.fit(X_train_subset, y_train)
            
            y_val_pred_proba = svc.predict_proba(X_val_subset)[:, 1]
            auc = roc_auc_score(y_val, y_val_pred_proba)
            
            auc_scores.append(auc)
            feature_subsets.append(current_features.copy())
            models.append(svc)
            
            if auc > best_auc:
                best_auc = auc
                best_features = current_features.copy()
                best_model = svc
        
        rfe_curve = {
            'feature_counts': list(range(max_features_to_select, len(auc_scores) * -self.rfe_step + max_features_to_select, -self.rfe_step)),
            'auc_scores': auc_scores
        }
        
        return best_model, best_features, best_auc, rfe_curve
    
    def train_models_with_adaptive_rfe(self, X_train, y_train, X_test, y_test):
        n_features = X_train.shape[1]
        
        test_predictions = np.zeros((X_test.shape[0], self.n_models))
        test_binary_predictions = np.zeros((X_test.shape[0], self.n_models))
        
        for i in tqdm(range(self.n_models), desc="Training models"):
            X_subsample, y_subsample = self.balanced_bootstrap_sampling(X_train, y_train)
            svc_model, selected_features, auc_score, rfe_curve = self.adaptive_rfe_for_single_model(
                X_subsample, y_subsample, X_test, y_test
            )
            
            if len(selected_features) > 0:
                X_test_selected = X_test[:, selected_features]
                y_test_pred_proba = svc_model.predict_proba(X_test_selected)[:, 1]
                test_predictions[:, i] = y_test_pred_proba
                y_test_pred_binary = (y_test_pred_proba > self.voting_threshold).astype(int)
                test_binary_predictions[:, i] = y_test_pred_binary
            else:
                test_predictions[:, i] = 0.5
                test_binary_predictions[:, i] = 0.5
                auc_score = 0.5
            
            self.models.append({
                'model': svc_model,
                'selected_features': selected_features,
                'test_auc': auc_score,
                'num_features': len(selected_features)
            })
            
            self.feature_subsets.append(selected_features)
            self.feature_counts.append(len(selected_features))
            self.rfe_auc_curves.append(rfe_curve)
            self.model_performances.append(auc_score)
        
        return test_predictions, test_binary_predictions
    
    def ensemble_predictions_with_hard_voting(self, test_binary_predictions):
        self.selected_model_indices = [i for i, auc in enumerate(self.model_performances) if auc > self.auc_threshold]
        self.n_selected_models = len(self.selected_model_indices)
        
        if self.n_selected_models == 0:
            self.selected_model_indices = list(range(self.n_models))
            self.n_selected_models = self.n_models
        
        selected_binary_predictions = test_binary_predictions[:, self.selected_model_indices]
        vote_counts = np.sum(selected_binary_predictions, axis=1)
        ensemble_pred = vote_counts / self.n_selected_models
        
        return ensemble_pred
    
    def fit(self, X_train, y_train, X_test, y_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        test_predictions, test_binary_predictions = self.train_models_with_adaptive_rfe(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        ensemble_pred = self.ensemble_predictions_with_hard_voting(test_binary_predictions)
        
        fpr, tpr, thresholds = roc_curve(y_test, ensemble_pred)
        youden_idx = tpr - fpr
        optimal_idx = np.argmax(youden_idx)
        optimal_threshold = thresholds[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]
        youden_value = youden_idx[optimal_idx]
        
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        ensemble_pred_binary = (ensemble_pred > optimal_threshold).astype(int)
        ensemble_acc = accuracy_score(y_test, ensemble_pred_binary)
        ensemble_f1 = f1_score(y_test, ensemble_pred_binary)
        
        print(f"\nEnsemble model performance on test set:")
        print(f"  AUC: {ensemble_auc:.4f}")
        print(f"  Accuracy: {ensemble_acc:.4f}")
        print(f"  F1-score: {ensemble_f1:.4f}")
        print(f"  Youden's Index: {youden_value:.4f}")
        
        if self.model_performances:
            avg_model_auc = np.mean(self.model_performances)
            best_model_auc = np.max(self.model_performances)
            avg_features = np.mean(self.feature_counts)
            
            if self.n_selected_models > 0:
                selected_model_performances = [self.model_performances[i] for i in self.selected_model_indices]
                avg_selected_auc = np.mean(selected_model_performances)
            else:
                avg_selected_auc = avg_model_auc
            
            print(f"\nIndividual model statistics:")
            print(f"  Total models trained: {len(self.models)}")
            print(f"  Models selected by AUC threshold (> {self.auc_threshold}): {self.n_selected_models}")
            print(f"  Average AUC of all models: {avg_model_auc:.4f}")
            print(f"  Average AUC of selected models: {avg_selected_auc:.4f}")
            print(f"  Best AUC: {best_model_auc:.4f}")
            print(f"  Average features selected: {avg_features:.1f}")
            print(f"  Ensemble improvement: {ensemble_auc - avg_model_auc:.4f}")
        
        # 创建验证集预测结果的DataFrame
        val_pred_df = pd.DataFrame({
            'label': y_test,
            'vote': ensemble_pred
        })
        
        return ensemble_pred, y_test, youden_value, val_pred_df
    
    def predict(self, X):
        if not self.models:
            raise ValueError("Model not trained, call fit first")
        
        X_scaled = self.scaler.transform(X)
        
        n_samples = X.shape[0]
        model_binary_predictions = np.zeros((n_samples, self.n_selected_models))
        
        for j, i in enumerate(self.selected_model_indices):
            model_info = self.models[i]
            model = model_info['model']
            selected_features = model_info['selected_features']
            
            if len(selected_features) > 0:
                X_selected = X_scaled[:, selected_features]
                y_pred_proba = model.predict_proba(X_selected)[:, 1]
                y_pred_binary = (y_pred_proba > self.voting_threshold).astype(int)
                model_binary_predictions[:, j] = y_pred_binary
            else:
                model_binary_predictions[:, j] = 0.5
        
        vote_counts = np.sum(model_binary_predictions, axis=1)
        ensemble_pred = vote_counts / self.n_selected_models
        
        return ensemble_pred
    
    def get_model_summary(self):
        if len(self.selected_model_indices) > 0:
            selected_aucs = [self.model_performances[i] for i in self.selected_model_indices]
            avg_auc_selected = np.mean(selected_aucs) if selected_aucs else 0
        else:
            avg_auc_selected = 0
        
        summary = {
            'n_models_total': len(self.models),
            'n_models_selected': self.n_selected_models,
            'selection_ratio': self.n_selected_models / len(self.models) if len(self.models) > 0 else 0,
            'avg_auc': np.mean(self.model_performances) if self.model_performances else 0,
            'avg_auc_selected': avg_auc_selected,
            'best_auc': np.max(self.model_performances) if self.model_performances else 0,
            'avg_features': np.mean(self.feature_counts) if self.feature_counts else 0,
            'min_features': np.min(self.feature_counts) if self.feature_counts else 0,
            'max_features': np.max(self.feature_counts) if self.feature_counts else 0,
            'auc_threshold': self.auc_threshold
        }
        
        return summary