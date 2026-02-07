"""
LASSO Feature Selection Script
Performs LASSO-based feature selection with cross-validation
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
import sys
import os

def lasso_feature_selection(X, y, cv=5, n_alphas=100, max_iter=10000, random_state=42):
    """
    Perform LASSO feature selection with cross-validation
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : array-like
        Target variable
    cv : int
        Number of cross-validation folds
    n_alphas : int
        Number of alpha values to test
    max_iter : int
        Maximum iterations for convergence
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    lasso_model : LassoCV
        Fitted LASSO model
    """
    print(f"Starting LASSO feature selection...")
    print(f"Input features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LASSO with cross-validation
    lasso = LassoCV(
        cv=cv,
        n_alphas=n_alphas,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit LASSO
    lasso.fit(X_scaled, y)
    
    # Get selected features (non-zero coefficients)
    coefficients = lasso.coef_
    selected_mask = np.abs(coefficients) > 0
    selected_features = X.columns[selected_mask].tolist()
    
    print(f"\nLASSO Results:")
    print(f"Optimal alpha: {lasso.alpha_:.6f}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Feature reduction: {X.shape[1]} -> {len(selected_features)}")
    
    return selected_features, lasso, scaler

def save_selected_features(selected_features, coefficients, output_path):
    """Save selected features and their coefficients"""
    df_features = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': coefficients[coefficients != 0]
    })
    df_features = df_features.sort_values('Coefficient', key=abs, ascending=False)
    df_features.to_csv(output_path, index=False)
    print(f"\nSelected features saved to: {output_path}")
    return df_features

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python feature_selection_lasso.py <feature_csv> <label_csv> [output_dir]")
        print("Example: python feature_selection_lasso.py features.csv labels.csv output/")
        sys.exit(1)
    
    # Input parameters
    feature_csv = sys.argv[1]
    label_csv = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'lasso_results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df_features = pd.read_csv(feature_csv)
    df_labels = pd.read_csv(label_csv)
    
    # Merge on ID
    if 'ID' in df_features.columns and 'ID' in df_labels.columns:
        df_merged = df_features.merge(df_labels, on='ID')
        X = df_merged.drop(['ID', 'label'], axis=1, errors='ignore')
        y = df_merged['label']
        ids = df_merged['ID']
    else:
        X = df_features.iloc[:, 1:]  # Assume first column is ID
        y = df_labels.iloc[:, 1]     # Assume second column is label
        ids = df_features.iloc[:, 0]
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    
    # Remove constant and near-constant features
    variance_threshold = 0.01
    feature_variance = X.var()
    valid_features = feature_variance[feature_variance > variance_threshold].index
    X = X[valid_features]
    print(f"After variance filtering: {X.shape[1]} features")
    
    # Perform LASSO feature selection
    selected_features, lasso_model, scaler = lasso_feature_selection(
        X, y, 
        cv=5, 
        n_alphas=100,
        random_state=42
    )
    
    # Save results
    feature_output = os.path.join(output_dir, 'selected_features_lasso.csv')
    df_selected = save_selected_features(
        selected_features, 
        lasso_model.coef_, 
        feature_output
    )
    
    # Save filtered dataset with selected features only
    X_selected = X[selected_features]
    df_output = pd.DataFrame(X_selected)
    df_output.insert(0, 'ID', ids)
    df_output['label'] = y.values
    
    output_csv = os.path.join(output_dir, 'features_lasso_selected.csv')
    df_output.to_csv(output_csv, index=False)
    print(f"Filtered dataset saved to: {output_csv}")
    
    # Save model and scaler
    import pickle
    model_path = os.path.join(output_dir, 'lasso_model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(lasso_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print("\nLASSO feature selection completed successfully!")
