"""
Comprehensive Benchmark: Ridge vs LASSO vs Adaptive LASSO
Compares three regression methods on House Prices dataset
"""

import numpy as np
import json
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import HousePriceDataLoader
from src.optimizer import AdaptiveLassoOptimizer, StandardLasso
from src.visualization import LassoVisualizer


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for regression methods.
    """
    
    def __init__(self, data_path='data/train.csv', output_dir='results'):
        """
        Initialize benchmark suite.
        
        Parameters:
        -----------
        data_path : str
            Path to House Prices dataset
        output_dir : str
            Directory for results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def run_benchmark(self):
        """
        Run complete benchmark pipeline.
        """
        print("\n" + "=" * 80)
        print("LASSO BENCHMARK SUITE")
        print("=" * 80)
        print("Theme 3: Dynamic Soft-Thresholding for Feature Selection")
        print("=" * 80 + "\n")
        
        # Step 1: Load data
        print("[1/6] Loading and preprocessing data...")
        loader = HousePriceDataLoader(filepath=self.data_path)
        X_train, X_test, y_train, y_test = loader.load_and_preprocess()
        feature_names = loader.get_feature_names()
        correlated_pairs = loader.get_correlated_pairs()
        
        # Step 2: Train Ridge regression
        print("\n[2/6] Training Ridge Regression (baseline)...")
        ridge_model, ridge_time = self._train_ridge(X_train, y_train)
        
        # Step 3: Train Standard LASSO
        print("\n[3/6] Training Standard LASSO (fixed λ)...")
        lasso_model, lasso_time = self._train_standard_lasso(X_train, y_train)
        
        # Step 4: Train Adaptive LASSO
        print("\n[4/6] Training Adaptive LASSO (dynamic λ)...")
        adaptive_model, adaptive_time = self._train_adaptive_lasso(X_train, y_train)
        
        # Step 5: Evaluate models
        print("\n[5/6] Evaluating models on test set...")
        self._evaluate_models({
            'Ridge': ridge_model,
            'Standard LASSO': lasso_model,
            'Adaptive LASSO': adaptive_model
        }, X_test, y_test, loader, {
            'Ridge': ridge_time,
            'Standard LASSO': lasso_time,
            'Adaptive LASSO': adaptive_time
        })
        
        # Step 6: Generate visualizations
        print("\n[6/6] Generating visualizations...")
        self._generate_visualizations({
            'Standard LASSO': lasso_model,
            'Adaptive LASSO': adaptive_model
        }, feature_names, correlated_pairs, X_test, y_test)
        
        # Save results
        self._save_results()
        
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE!")
        print("=" * 80)
        print(f"✓ Results saved to: {self.output_dir}/")
        print(f"✓ Performance metrics: {self.output_dir}/performance_metrics.json")
        print(f"✓ Sparsity table: {self.output_dir}/sparsity_table.csv")
        print(f"✓ Visualizations: {self.output_dir}/*.png")
        print("=" * 80 + "\n")
    
    def _train_ridge(self, X_train, y_train):
        """Train Ridge regression."""
        start_time = time.time()
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        return model, training_time
    
    def _train_standard_lasso(self, X_train, y_train):
        """Train standard LASSO."""
        start_time = time.time()
        model = StandardLasso(
            lambda_val=0.01,
            max_iter=1500,
            learning_rate=0.01,
            tol=1e-6,
            verbose=True
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        return model, training_time
    
    def _train_adaptive_lasso(self, X_train, y_train):
        """Train adaptive LASSO."""
        start_time = time.time()
        model = AdaptiveLassoOptimizer(
            lambda_0=0.015,
            alpha=0.08,
            max_iter=1500,
            learning_rate=0.01,
            tol=1e-6,
            verbose=True
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        return model, training_time
    
    def _evaluate_models(self, models_dict, X_test, y_test, loader, training_times):
        """Evaluate all models and store results."""
        print("\n" + "-" * 80)
        print("MODEL EVALUATION RESULTS")
        print("-" * 80)
        
        for model_name, model in models_dict.items():
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Sparsity
            coefs = model.coef_
            n_features = len(coefs)
            n_zero = np.sum(np.abs(coefs) < 1e-6)
            sparsity_pct = 100 * n_zero / n_features
            
            # Iterations (if available)
            n_iter = model.n_iter_ if hasattr(model, 'n_iter_') else 'N/A'
            
            # Store results
            self.results[model_name] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'sparsity_percent': float(sparsity_pct),
                'n_nonzero_coefs': int(n_features - n_zero),
                'n_zero_coefs': int(n_zero),
                'total_features': int(n_features),
                'iterations': n_iter if n_iter != 'N/A' else n_iter,
                'training_time_seconds': float(training_times[model_name])
            }
            
            # Print summary
            print(f"\n{model_name}:")
            print(f"  MSE:                {mse:.6f}")
            print(f"  RMSE:               {rmse:.6f}")
            print(f"  MAE:                {mae:.6f}")
            print(f"  R²:                 {r2:.6f}")
            print(f"  Sparsity:           {sparsity_pct:.2f}%")
            print(f"  Non-zero features:  {n_features - n_zero}/{n_features}")
            print(f"  Iterations:         {n_iter}")
            print(f"  Training time:      {training_times[model_name]:.2f}s")
        
        print("-" * 80)
        
        # Compute improvements
        if 'Adaptive LASSO' in self.results and 'Standard LASSO' in self.results:
            print("\nADAPTIVE LASSO IMPROVEMENTS OVER STANDARD LASSO:")
            print("-" * 80)
            
            adaptive = self.results['Adaptive LASSO']
            standard = self.results['Standard LASSO']
            
            mse_improvement = 100 * (standard['mse'] - adaptive['mse']) / standard['mse']
            sparsity_improvement = adaptive['sparsity_percent'] - standard['sparsity_percent']
            
            if standard['iterations'] != 'N/A' and adaptive['iterations'] != 'N/A':
                iter_reduction = 100 * (standard['iterations'] - adaptive['iterations']) / standard['iterations']
                print(f"  Iterations reduced:     {iter_reduction:+.1f}%")
            
            print(f"  MSE improvement:        {mse_improvement:+.2f}%")
            print(f"  Sparsity improvement:   {sparsity_improvement:+.1f} percentage points")
            print(f"  Time saved:             {standard['training_time_seconds'] - adaptive['training_time_seconds']:.2f}s")
            print("-" * 80)
    
    def _generate_visualizations(self, models_dict, feature_names, 
                                correlated_pairs, X_test, y_test):
        """Generate all visualizations."""
        visualizer = LassoVisualizer(output_dir=self.output_dir)
        
        # 1. Coefficient paths (THE WOW PLOT!)
        print("\n  → Generating coefficient path plot...")
        visualizer.plot_coefficient_paths(models_dict, feature_names, top_k=30)
        
        # 2. Convergence comparison
        print("  → Generating convergence comparison...")
        visualizer.plot_convergence_comparison(models_dict)
        
        # 3. Feature importance (for Adaptive LASSO)
        print("  → Generating feature importance plot...")
        adaptive_model = models_dict['Adaptive LASSO']
        visualizer.plot_feature_importance(adaptive_model, feature_names, top_k=30,
                                          save_path='feature_importance_adaptive.png')
        
        # 4. Predictions vs actual
        print("  → Generating predictions comparison...")
        y_pred_dict = {
            name: model.predict(X_test) 
            for name, model in models_dict.items()
        }
        visualizer.plot_predictions_vs_actual(y_test, y_pred_dict)
        
        # 5. Multicollinearity analysis
        if len(correlated_pairs) > 0:
            print("  → Generating multicollinearity analysis...")
            visualizer.plot_multicollinearity_analysis(
                correlated_pairs, models_dict, feature_names
            )
    
    def _save_results(self):
        """Save results to files."""
        # Save JSON metrics
        with open(f'{self.output_dir}/performance_metrics.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save sparsity table as CSV
        sparsity_data = []
        for model_name, metrics in self.results.items():
            sparsity_data.append({
                'Method': model_name,
                'MSE': f"{metrics['mse']:.6f}",
                'RMSE': f"{metrics['rmse']:.6f}",
                'R²': f"{metrics['r2']:.6f}",
                'Sparsity (%)': f"{metrics['sparsity_percent']:.2f}",
                'Non-zero Features': f"{metrics['n_nonzero_coefs']}/{metrics['total_features']}",
                'Iterations': metrics['iterations'],
                'Training Time (s)': f"{metrics['training_time_seconds']:.2f}"
            })
        
        df = pd.DataFrame(sparsity_data)
        df.to_csv(f'{self.output_dir}/sparsity_table.csv', index=False)
        
        print(f"\n✓ Performance metrics saved")


def main():
    """Main execution."""
    # Check if dataset exists
    data_path = 'data/train.csv'
    
    if not os.path.exists(data_path):
        print("\n" + "=" * 80)
        print("ERROR: Dataset not found!")
        print("=" * 80)
        print(f"Please download the House Prices dataset from Kaggle:")
        print("https://www.kaggle.com/c/house-prices-advanced-regression-techniques")
        print(f"\nPlace 'train.csv' in the 'data/' folder")
        print("=" * 80 + "\n")
        return
    
    # Run benchmark
    benchmark = BenchmarkSuite(data_path=data_path)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
