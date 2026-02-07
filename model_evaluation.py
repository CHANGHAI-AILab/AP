"""
Model Evaluation Script
Generates ROC curves, calibration plots, and comprehensive performance metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import pickle
import sys
import os
import warnings
warnings.filterwarnings('ignore')

def plot_roc_curve(y_true, y_pred_proba, output_path, title="ROC Curve"):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {output_path}")
    
    return fpr, tpr, roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba, output_path, title="Precision-Recall Curve"):
    """Plot and save Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to: {output_path}")
    
    return precision, recall, avg_precision

def plot_calibration_curve(y_true, y_pred_proba, output_path, n_bins=10, title="Calibration Curve"):
    """Plot and save calibration curve"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
             label='Model', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], "k--", label='Perfect calibration', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration curve saved to: {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()
    
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")
    
    return cm

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive performance metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'average_precision': average_precision_score(y_true, y_pred_proba)
    }
    
    return metrics

def evaluate_model_comprehensive(model_path, scaler_path, data_csv, output_dir):
    """Comprehensive model evaluation"""
    print("Loading model and data...")
    
    # Load model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load data
    df = pd.read_csv(data_csv)
    
    if 'label' in df.columns:
        y_true = df['label'].values
        X = df.drop(['ID', 'label'], axis=1, errors='ignore').values
    else:
        print("Error: 'label' column not found")
        return
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print("\nPerformance Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # ROC curve
    plot_roc_curve(y_true, y_pred_proba, 
                   os.path.join(output_dir, 'roc_curve.png'))
    
    # Precision-Recall curve
    plot_precision_recall_curve(y_true, y_pred_proba,
                                os.path.join(output_dir, 'pr_curve.png'))
    
    # Calibration curve
    plot_calibration_curve(y_true, y_pred_proba,
                          os.path.join(output_dir, 'calibration_curve.png'))
    
    # Confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred,
                               os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save classification report
    report = classification_report(y_true, y_pred, 
                                   target_names=['Negative', 'Positive'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to: {report_path}")
    
    print("\nEvaluation completed successfully!")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python model_evaluation.py <model_path> <scaler_path> <data_csv> [output_dir]")
        print("Example: python model_evaluation.py trained_models/dlr_model.pkl trained_models/dlr_scaler.pkl test_data.csv evaluation_results/")
        sys.exit(1)
    
    model_path = sys.argv[1]
    scaler_path = sys.argv[2]
    data_csv = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else 'evaluation_results'
    
    evaluate_model_comprehensive(model_path, scaler_path, data_csv, output_dir)
