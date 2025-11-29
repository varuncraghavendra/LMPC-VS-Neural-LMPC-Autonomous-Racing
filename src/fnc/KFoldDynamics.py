"""
Improved K-Fold with Random Forest (Nonlinear!)
Now uses Random Forest instead of Ridge regression
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

class KFoldDynamicsPredictor:
    """K-Fold with Random Forest base models"""
    def __init__(self, state_dim=6, input_dim=2, n_folds=5):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.n_folds = n_folds
        
        self.all_states = []
        self.all_actions = []
        self.all_next_states = []
        
        self.fold_models = []
        self.fold_scores = []
        
        self.X_mean = None
        self.X_std = None
        
        print(f"✓ Initialized K-Fold with Random Forest (K={n_folds})")
    
    def add_trajectory(self, states, actions):
        """Add trajectory data"""
        for t in range(len(states) - 1):
            self.all_states.append(states[t])
            self.all_actions.append(actions[t])
            self.all_next_states.append(states[t + 1])
    
    def train(self, verbose=True):
        """Train K-Fold with Random Forest"""
        if len(self.all_states) == 0:
            print("⚠ No training data!")
            return
        
        states = np.array(self.all_states)
        actions = np.array(self.all_actions)
        next_states = np.array(self.all_next_states)
        
        X = np.hstack([states, actions])
        
        # Normalize
        if self.X_mean is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
        
        X_norm = (X - self.X_mean) / self.X_std
        
        if verbose:
            print(f"\nK-Fold CV with Random Forest")
            print(f"Samples: {X.shape[0]}, Folds: {self.n_folds}")
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        self.fold_models = []
        self.fold_scores = []
        
        state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_norm)):
            if verbose:
                print(f"\n  Fold {fold_idx + 1}/{self.n_folds}:")
            
            X_train, X_val = X_norm[train_idx], X_norm[val_idx]
            y_train, y_val = next_states[train_idx], next_states[val_idx]
            
            models_this_fold = []
            scores_this_fold = []
            
            for state_idx in range(self.state_dim):
                # Random Forest (NONLINEAR!)
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train[:, state_idx])
                score = model.score(X_val, y_val[:, state_idx])
                scores_this_fold.append(score)
                models_this_fold.append(model)
                
                if verbose:
                    print(f"    {state_names[state_idx]}: R²={score:.4f}")
            
            self.fold_models.append(models_this_fold)
            avg_score = np.mean(scores_this_fold)
            self.fold_scores.append(avg_score)
            
            if verbose:
                print(f"    Average: R²={avg_score:.4f}")
        
        if verbose:
            overall = np.mean(self.fold_scores)
            print(f"\n✓ K-Fold complete! Average R²: {overall:.4f}\n")
    
    def predict(self, state, action, return_std=True):
        """Predict using ensemble"""
        X = np.hstack([state, action]).reshape(1, -1)
        
        if self.X_mean is not None:
            X = (X - self.X_mean) / self.X_std
        
        all_predictions = []
        
        for fold_models in self.fold_models:
            fold_pred = []
            for model in fold_models:
                pred = model.predict(X)[0]
                fold_pred.append(pred)
            all_predictions.append(fold_pred)
        
        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        
        if return_std:
            std_pred = np.std(all_predictions, axis=0)
            return mean_pred, std_pred
        else:
            return mean_pred
    
    def save_model(self, filepath):
        """Save models"""
        data = {
            'fold_models': self.fold_models,
            'fold_scores': self.fold_scores,
            'state_dim': self.state_dim,
            'input_dim': self.input_dim,
            'n_folds': self.n_folds,
            'X_mean': self.X_mean,
            'X_std': self.X_std
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ K-Fold saved to {filepath}")
    
    def load_model(self, filepath):
        """Load models"""
        if not os.path.exists(filepath):
            print(f"⚠ File not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.fold_models = data['fold_models']
        self.fold_scores = data['fold_scores']
        self.state_dim = data['state_dim']
        self.input_dim = data['input_dim']
        self.n_folds = data['n_folds']
        self.X_mean = data.get('X_mean', None)
        self.X_std = data.get('X_std', None)
        
        print(f"✓ K-Fold loaded from {filepath}")
