"""
Deep Learning Radiomics (DLR) Model Training Script
Trains logistic regression model with hyperparameter tuning and cross-validation
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pickle

def train_dlr_model(X_train, y_train, cv=5, random_state=42):
    """
    Train DLR (Deep Learning Radiomics) logistic regression model with hyperparameter tuning
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
        
    Returns:
    --------
    best_model : LogisticRegression
        Best fitted model
    grid_search : GridSearchCV
        Grid search object with all results
    """
    print("Starting hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 5000, 10000]
    }
    
    # Base model
    base_model = LogisticRegression(
        random_state=random_state,
        class_weight='balanced'
    )
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search

def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Evaluate model performance"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    auc = roc_auc_score(y, y_pred_proba)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n{dataset_name} Performance:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def save_cv_results(grid_search, output_path):
    """Save cross-validation results"""
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values('rank_test_score')
    cv_results.to_csv(output_path, index=False)
    print(f"CV results saved to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python model_training_dlr.py <feature_csv> [output_dir] [test_size]")
        print("Example: python model_training_dlr.py features_lasso_selected.csv models/ 0.2")
        sys.exit(1)
    
    # Input parameters
    feature_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'trained_models'
    test_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(feature_csv)
    
    # Separate features and labels
    if 'label' in df.columns:
        y = df['label']
        X = df.drop(['ID', 'label'], axis=1, errors='ignore')
        ids = df['ID'] if 'ID' in df.columns else None
    else:
        print("Error: 'label' column not found in dataset")
        sys.exit(1)
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )
    
    if ids is not None:
        ids_train, ids_test = train_test_split(
            ids, test_size=test_size, random_state=42, stratify=y
        )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with hyperparameter tuning
    best_model, grid_search = train_dlr_model(
        X_train_scaled, 
        y_train, 
        cv=5, 
        random_state=42
    )
    
    # Evaluate on train and test sets
    train_results = evaluate_model(best_model, X_train_scaled, y_train, "Training")
    test_results = evaluate_model(best_model, X_test_scaled, y_test, "Test")
    
    # Save model and scaler
    model_path = os.path.join(output_dir, 'dlr_model.pkl')
    scaler_path = os.path.join(output_dir, 'dlr_scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Save CV results
    cv_results_path = os.path.join(output_dir, 'cv_results.csv')
    save_cv_results(grid_search, cv_results_path)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'ID': ids_test.values if ids is not None else range(len(y_test)),
        'True_Label': y_test.values,
        'Predicted_Label': test_results['predictions'],
        'Predicted_Probability': test_results['probabilities']
    })
    predictions_path = os.path.join(output_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")
    
    # Save performance summary
    summary = {
        'train_auc': train_results['auc'],
        'train_accuracy': train_results['accuracy'],
        'train_f1': train_results['f1_score'],
        'test_auc': test_results['auc'],
        'test_accuracy': test_results['accuracy'],
        'test_f1': test_results['f1_score'],
        'best_params': str(grid_search.best_params_),
        'n_features': X.shape[1],
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0]
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, 'model_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    print("\nModel training completed successfully!")
