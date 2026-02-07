"""
Complete Pipeline: Feature Extraction to Model Training
Integrates all steps: feature extraction, LASSO selection, model training, and evaluation
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

class RadiomicsPipeline:
    """Complete radiomics analysis pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def log(self, message):
        """Print timestamped log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def step1_feature_extraction(self):
        """Step 1: Extract radiomics features"""
        self.log("=" * 60)
        self.log("STEP 1: Feature Extraction")
        self.log("=" * 60)
        
        image_dir = self.config['image_dir']
        mask_dir = self.config['mask_dir']
        output_dir = self.config['feature_output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract 3D radiomics features
        self.log("Extracting 3D radiomics features...")
        cmd = f"python 3d_Radiomics_feature.py {image_dir} {mask_dir}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Extract intratumoral and peritumoral features
        if self.config.get('extract_peritumoral', False):
            self.log("Creating peritumoral regions...")
            mask_p2_dir = mask_dir + "_p2"
            mask_p5_dir = mask_dir + "_p5"
            
            cmd = f"python Intratumoral_peritumoral_feature.py {mask_dir} {mask_p2_dir} 2"
            subprocess.run(cmd, shell=True, check=True)
            
            cmd = f"python Intratumoral_peritumoral_feature.py {mask_dir} {mask_p5_dir} 5"
            subprocess.run(cmd, shell=True, check=True)
            
            # Extract features from peritumoral regions
            self.log("Extracting peritumoral features...")
            cmd = f"python 3d_Radiomics_feature.py {image_dir} {mask_p2_dir}"
            subprocess.run(cmd, shell=True, check=True)
            
            cmd = f"python 3d_Radiomics_feature.py {image_dir} {mask_p5_dir}"
            subprocess.run(cmd, shell=True, check=True)
        
        # Extract deep learning features
        if self.config.get('extract_dl_features', False):
            self.log("Extracting ResNet18 features...")
            max_roi_dir = self.config.get('max_roi_dir', image_dir + "_max_roi")
            cmd = f"python max_roi_resnet18_radio_feature.py {max_roi_dir} 2d"
            subprocess.run(cmd, shell=True, check=True)
            
            self.log("Extracting DINOv2 features...")
            cmd = f"python dinov2_feature.py {max_roi_dir}"
            subprocess.run(cmd, shell=True, check=True)
        
        self.log("Feature extraction completed!")
        
    def step2_concatenate_features(self):
        """Step 2: Concatenate all features"""
        self.log("=" * 60)
        self.log("STEP 2: Feature Concatenation")
        self.log("=" * 60)
        
        # Run concatenation script
        self.log("Concatenating all features...")
        subprocess.run("python concat_all_features.py", shell=True, check=True)
        
        self.log("Feature concatenation completed!")
        
    def step3_lasso_selection(self):
        """Step 3: LASSO feature selection"""
        self.log("=" * 60)
        self.log("STEP 3: LASSO Feature Selection")
        self.log("=" * 60)
        
        feature_csv = self.config['concatenated_features']
        label_csv = self.config['label_csv']
        output_dir = self.config['lasso_output_dir']
        
        self.log(f"Input features: {feature_csv}")
        self.log(f"Labels: {label_csv}")
        
        cmd = f"python feature_selection_lasso.py {feature_csv} {label_csv} {output_dir}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Store selected features path
        self.results['selected_features'] = os.path.join(output_dir, 'features_lasso_selected.csv')
        
        self.log("LASSO feature selection completed!")
        
    def step4_model_training(self):
        """Step 4: Train DLR model"""
        self.log("=" * 60)
        self.log("STEP 4: Model Training")
        self.log("=" * 60)
        
        feature_csv = self.results['selected_features']
        output_dir = self.config['model_output_dir']
        test_size = self.config.get('test_size', 0.2)
        
        self.log(f"Training on: {feature_csv}")
        
        cmd = f"python model_training_dlr.py {feature_csv} {output_dir} {test_size}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Store model paths
        self.results['model_path'] = os.path.join(output_dir, 'dlr_model.pkl')
        self.results['scaler_path'] = os.path.join(output_dir, 'dlr_scaler.pkl')
        
        self.log("Model training completed!")
        
    def step5_model_evaluation(self):
        """Step 5: Evaluate model"""
        self.log("=" * 60)
        self.log("STEP 5: Model Evaluation")
        self.log("=" * 60)
        
        model_path = self.results['model_path']
        scaler_path = self.results['scaler_path']
        
        # Evaluate on test set
        test_csv = self.config.get('test_csv', self.results['selected_features'])
        output_dir = self.config['evaluation_output_dir']
        
        self.log(f"Evaluating on: {test_csv}")
        
        cmd = f"python model_evaluation.py {model_path} {scaler_path} {test_csv} {output_dir}"
        subprocess.run(cmd, shell=True, check=True)
        
        self.log("Model evaluation completed!")
        
    def run_full_pipeline(self):
        """Run complete pipeline"""
        self.log("=" * 60)
        self.log("STARTING FULL RADIOMICS PIPELINE")
        self.log("=" * 60)
        
        start_time = datetime.now()
        
        try:
            if self.config.get('run_feature_extraction', True):
                self.step1_feature_extraction()
            
            if self.config.get('run_concatenation', True):
                self.step2_concatenate_features()
            
            if self.config.get('run_lasso', True):
                self.step3_lasso_selection()
            
            if self.config.get('run_training', True):
                self.step4_model_training()
            
            if self.config.get('run_evaluation', True):
                self.step5_model_evaluation()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.log("=" * 60)
            self.log("PIPELINE COMPLETED SUCCESSFULLY!")
            self.log(f"Total duration: {duration}")
            self.log("=" * 60)
            
        except Exception as e:
            self.log(f"ERROR: Pipeline failed with error: {str(e)}")
            raise

def load_config(config_file):
    """Load configuration from file"""
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    # Example configuration
    default_config = {
        # Input directories
        'image_dir': 'data/images',
        'mask_dir': 'data/masks',
        'label_csv': 'data/labels.csv',
        
        # Output directories
        'feature_output_dir': 'output/features',
        'lasso_output_dir': 'output/lasso_results',
        'model_output_dir': 'output/trained_models',
        'evaluation_output_dir': 'output/evaluation',
        
        # Feature extraction options
        'extract_peritumoral': True,
        'extract_dl_features': True,
        
        # Model training options
        'test_size': 0.2,
        
        # Pipeline steps to run
        'run_feature_extraction': True,
        'run_concatenation': True,
        'run_lasso': True,
        'run_training': True,
        'run_evaluation': True,
        
        # Paths for intermediate files
        'concatenated_features': 'output/features/all_features.csv',
        'max_roi_dir': 'data/max_roi_images'
    }
    
    if len(sys.argv) > 1:
        # Load config from file
        config_file = sys.argv[1]
        print(f"Loading configuration from: {config_file}")
        config = load_config(config_file)
    else:
        print("No config file provided, using default configuration")
        print("Usage: python pipeline_full.py <config.json>")
        print("\nTo create a config file, save the following template as config.json:")
        import json
        print(json.dumps(default_config, indent=2))
        print("\nProceeding with default config...")
        config = default_config
    
    # Run pipeline
    pipeline = RadiomicsPipeline(config)
    pipeline.run_full_pipeline()
