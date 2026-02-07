"""
Test script to verify the pipeline components work correctly
Creates synthetic data and runs through the complete pipeline
"""
import numpy as np
import pandas as pd
import os
import shutil

def create_synthetic_data(n_samples=100, n_features=50, output_dir='test_data'):
    """Create synthetic dataset for testing"""
    print("Creating synthetic test data...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random features
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (binary classification)
    # Make some features predictive
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5) > 0
    y = y.astype(int)
    
    # Create feature dataframe
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df_features = pd.DataFrame(X, columns=feature_names)
    df_features.insert(0, 'ID', [f'patient_{i:03d}' for i in range(n_samples)])
    
    # Create label dataframe
    df_labels = pd.DataFrame({
        'ID': [f'patient_{i:03d}' for i in range(n_samples)],
        'label': y
    })
    
    # Save to CSV
    feature_path = os.path.join(output_dir, 'features.csv')
    label_path = os.path.join(output_dir, 'labels.csv')
    
    df_features.to_csv(feature_path, index=False)
    df_labels.to_csv(label_path, index=False)
    
    print(f"Created {n_samples} samples with {n_features} features")
    print(f"Label distribution: {np.bincount(y)}")
    print(f"Features saved to: {feature_path}")
    print(f"Labels saved to: {label_path}")
    
    return feature_path, label_path

def test_lasso_selection(feature_path, label_path):
    """Test LASSO feature selection"""
    print("\n" + "="*60)
    print("Testing LASSO Feature Selection")
    print("="*60)
    
    output_dir = 'test_output/lasso_results'
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"python feature_selection_lasso.py {feature_path} {label_path} {output_dir}"
    print(f"Running: {cmd}")
    
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ LASSO selection completed successfully")
        return os.path.join(output_dir, 'features_lasso_selected.csv')
    else:
        print("✗ LASSO selection failed")
        print(result.stderr)
        return None

def test_model_training(feature_path):
    """Test model training"""
    print("\n" + "="*60)
    print("Testing Model Training")
    print("="*60)
    
    output_dir = 'test_output/trained_models'
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"python model_training_dlr.py {feature_path} {output_dir} 0.2"
    print(f"Running: {cmd}")
    
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Model training completed successfully")
        model_path = os.path.join(output_dir, 'dlr_model.pkl')
        scaler_path = os.path.join(output_dir, 'dlr_scaler.pkl')
        return model_path, scaler_path
    else:
        print("✗ Model training failed")
        print(result.stderr)
        return None, None

def test_model_evaluation(model_path, scaler_path, data_path):
    """Test model evaluation"""
    print("\n" + "="*60)
    print("Testing Model Evaluation")
    print("="*60)
    
    output_dir = 'test_output/evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"python model_evaluation.py {model_path} {scaler_path} {data_path} {output_dir}"
    print(f"Running: {cmd}")
    
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Model evaluation completed successfully")
        return True
    else:
        print("✗ Model evaluation failed")
        print(result.stderr)
        return False

def verify_outputs():
    """Verify all expected outputs exist"""
    print("\n" + "="*60)
    print("Verifying Outputs")
    print("="*60)
    
    expected_files = [
        'test_output/lasso_results/selected_features_lasso.csv',
        'test_output/lasso_results/features_lasso_selected.csv',
        'test_output/lasso_results/lasso_model.pkl',
        'test_output/trained_models/dlr_model.pkl',
        'test_output/trained_models/dlr_scaler.pkl',
        'test_output/trained_models/cv_results.csv',
        'test_output/trained_models/model_summary.csv',
        'test_output/evaluation/roc_curve.png',
        'test_output/evaluation/pr_curve.png',
        'test_output/evaluation/calibration_curve.png',
        'test_output/evaluation/confusion_matrix.png',
        'test_output/evaluation/evaluation_metrics.csv'
    ]
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def cleanup():
    """Clean up test files"""
    print("\n" + "="*60)
    print("Cleanup")
    print("="*60)
    
    response = input("Do you want to delete test files? (y/n): ")
    if response.lower() == 'y':
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
            print("✓ Removed test_data/")
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
            print("✓ Removed test_output/")
        print("Cleanup completed")
    else:
        print("Test files preserved for inspection")

def main():
    """Run complete pipeline test"""
    print("="*60)
    print("PIPELINE TEST SUITE")
    print("="*60)
    
    try:
        # Create synthetic data
        feature_path, label_path = create_synthetic_data(
            n_samples=100, 
            n_features=50
        )
        
        # Test LASSO selection
        selected_features_path = test_lasso_selection(feature_path, label_path)
        if selected_features_path is None:
            print("\n✗ Pipeline test FAILED at LASSO selection")
            return
        
        # Test model training
        model_path, scaler_path = test_model_training(selected_features_path)
        if model_path is None or scaler_path is None:
            print("\n✗ Pipeline test FAILED at model training")
            return
        
        # Test model evaluation
        success = test_model_evaluation(model_path, scaler_path, selected_features_path)
        if not success:
            print("\n✗ Pipeline test FAILED at model evaluation")
            return
        
        # Verify all outputs
        all_outputs_exist = verify_outputs()
        
        # Final summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        if all_outputs_exist:
            print("✓ ALL TESTS PASSED")
            print("\nThe pipeline is working correctly!")
            print("All expected outputs were generated.")
        else:
            print("✗ SOME TESTS FAILED")
            print("\nSome expected outputs are missing.")
            print("Please check the error messages above.")
        
        # Cleanup
        cleanup()
        
    except Exception as e:
        print(f"\n✗ Pipeline test FAILED with exception:")
        print(str(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
