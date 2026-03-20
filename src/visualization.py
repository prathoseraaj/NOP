"""
Visualization Tools for LASSO Comparison
Creates compelling plots to showcase the advantages of Adaptive LASSO
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class LassoVisualizer:
    """
    Comprehensive visualization suite for LASSO comparison.
    """
    
    def __init__(self, output_dir='results'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        """
        self.output_dir = output_dir
    
    def plot_coefficient_paths(self, models_dict, feature_names=None, 
                               top_k=20, save_path='coefficient_path.png'):
        """
        The "WOW" visualization: Coefficient evolution over iterations.
        
        Shows how features are progressively pruned to zero.
        This is the visual proof that your math works!
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of model_name -> fitted_model
        feature_names : list, optional
            Names of features
        top_k : int
            Number of top features to show
        save_path : str
            Filename to save plot
        """
        n_models = len(models_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            ax = axes[idx]
            
            # Get coefficient history
            coef_history = np.array(model.coef_history_)  # Shape: (n_iter, n_features)
            
            # Select top-k features by peak absolute value across the whole trajectory.
            # This avoids selecting only near-zero lines at the final iteration.
            peak_coefs = np.max(np.abs(coef_history), axis=0)
            top_indices = np.argsort(peak_coefs)[-top_k:]
            
            # Plot each feature's trajectory
            iterations = np.arange(len(coef_history))
            for feat_idx in top_indices:
                trajectory = coef_history[:, feat_idx]
                
                # Color by final value (positive=blue, negative=red, near-zero=gray)
                final_val = coef_history[-1, feat_idx]
                if abs(final_val) < 1e-6:
                    color = 'gray'
                    alpha = 0.3
                    linewidth = 0.5
                else:
                    color = 'blue' if final_val > 0 else 'red'
                    alpha = 0.7
                    linewidth = 1.5
                
                ax.plot(iterations, trajectory, color=color, alpha=alpha, 
                       linewidth=linewidth)
            
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Coefficient Value (β)')
            ax.set_title(f'{model_name}\nCoefficient Paths (Top {top_k} Features)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{save_path}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved coefficient path plot: {self.output_dir}/{save_path}")
        plt.close()
    
    def plot_convergence_comparison(self, models_dict, save_path='convergence_comparison.png'):
        """
        Compare convergence speed across models.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of model_name -> fitted_model
        save_path : str
            Filename to save plot
        """
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig)
        
        # Plot 1: Loss over iterations
        ax1 = fig.add_subplot(gs[0, 0])
        for model_name, model in models_dict.items():
            loss_history = model.loss_history_
            iterations = np.arange(len(loss_history))
            ax1.plot(iterations, loss_history, label=model_name, linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Function Value')
        ax1.set_title('Loss Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Sparsity over iterations
        ax2 = fig.add_subplot(gs[0, 1])
        for model_name, model in models_dict.items():
            if hasattr(model, 'sparsity_history_'):
                sparsity_history = model.sparsity_history_
                iterations = np.arange(len(sparsity_history))
                ax2.plot(iterations, sparsity_history, label=model_name, linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Sparsity (%)')
        ax2.set_title('Feature Pruning Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Lambda evolution (for Adaptive LASSO)
        ax3 = fig.add_subplot(gs[0, 2])
        for model_name, model in models_dict.items():
            if hasattr(model, 'lambda_history_'):
                lambda_history = model.lambda_history_
                iterations = np.arange(len(lambda_history))
                ax3.plot(iterations, lambda_history, label=model_name, linewidth=2)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('λ (Regularization Strength)')
        ax3.set_title('Dynamic λ Evolution (Cooling Schedule)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{save_path}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved convergence comparison: {self.output_dir}/{save_path}")
        plt.close()
    
    def plot_feature_importance(self, model, feature_names=None, top_k=30,
                               save_path='feature_importance.png'):
        """
        Plot top-k important features by coefficient magnitude.
        
        Parameters:
        -----------
        model : fitted model
            Trained model with coef_ attribute
        feature_names : list, optional
            Names of features
        top_k : int
            Number of top features to display
        save_path : str
            Filename to save plot
        """
        importance = model.get_feature_importance(feature_names)
        
        # Filter non-zero coefficients and take top-k
        non_zero = [(name, coef, abs_coef) for name, coef, abs_coef in importance 
                    if abs_coef >= 1e-6]
        top_features = non_zero[:min(top_k, len(non_zero))]
        
        if len(top_features) == 0:
            print("⚠ No non-zero features to plot")
            return
        
        # Prepare data
        names = [f[:30] for f, _, _ in top_features]  # Truncate long names
        coefs = [c for _, c, _ in top_features]
        colors = ['blue' if c > 0 else 'red' for c in coefs]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.3)))
        y_pos = np.arange(len(names))
        
        ax.barh(y_pos, coefs, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Coefficient Value (β)')
        ax.set_title(f'Top {len(top_features)} Important Features\n(Non-zero Coefficients)')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{save_path}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance plot: {self.output_dir}/{save_path}")
        plt.close()
    
    def plot_predictions_vs_actual(self, y_true, y_pred_dict, 
                                   save_path='predictions_comparison.png'):
        """
        Scatter plot of predictions vs actual values for all models.
        
        Parameters:
        -----------
        y_true : array
            True target values
        y_pred_dict : dict
            Dictionary of model_name -> predictions
        save_path : str
            Filename to save plot
        """
        n_models = len(y_pred_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect Prediction')
            
            # Compute metrics
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / 
                     np.sum((y_true - y_true.mean()) ** 2))
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{model_name}\nRMSE: {rmse:.4f} | R²: {r2:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{save_path}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved predictions comparison: {self.output_dir}/{save_path}")
        plt.close()
    
    def plot_multicollinearity_analysis(self, correlated_pairs, models_dict, 
                                       feature_names, save_path='multicollinearity_analysis.png'):
        """
        Analyze how different models handle highly correlated features.
        
        Parameters:
        -----------
        correlated_pairs : list
            List of correlated feature pairs from data loader
        models_dict : dict
            Dictionary of model_name -> fitted_model
        feature_names : list
            Names of all features
        save_path : str
            Filename to save plot
        """
        if len(correlated_pairs) == 0:
            print("⚠ No correlated pairs to analyze")
            return
        
        # Take top correlated pairs that are actually present in feature_names
        candidate_pairs = sorted(correlated_pairs, key=lambda x: x['correlation'], reverse=True)
        top_pairs = [
            p for p in candidate_pairs
            if p['feature_1'] in feature_names and p['feature_2'] in feature_names
        ][:5]

        if len(top_pairs) == 0:
            print("⚠ No valid correlated pairs found in encoded feature set")
            return
        
        n_pairs = len(top_pairs)
        n_models = len(models_dict)
        
        fig, axes = plt.subplots(n_pairs, n_models, figsize=(5*n_models, 4*n_pairs))
        
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for pair_idx, pair in enumerate(top_pairs):
            feat1 = pair['feature_1']
            feat2 = pair['feature_2']
            corr = pair['correlation']
            
            # Get indices
            idx1 = feature_names.index(feat1)
            idx2 = feature_names.index(feat2)
            
            for model_idx, (model_name, model) in enumerate(models_dict.items()):
                ax = axes[pair_idx, model_idx]

                coef1 = model.coef_[idx1]
                coef2 = model.coef_[idx2]

                # Bar plot
                vals = [coef1, coef2]
                ax.bar([feat1[:20], feat2[:20]], vals,
                       color=['blue', 'orange'], alpha=0.7)
                ax.axhline(y=0, color='black', linewidth=0.8)
                ax.set_ylabel('Coefficient Value')
                ax.set_title(f'{model_name}\nρ = {corr:.3f}')
                ax.grid(True, alpha=0.3, axis='y')

                # Avoid visually empty subplots when coefficients are tiny
                max_abs = max(abs(coef1), abs(coef2), 1e-4)
                ax.set_ylim(-1.25 * max_abs, 1.25 * max_abs)

                # Show numeric values directly on bars
                for x_i, v in enumerate(vals):
                    ax.text(x_i, v, f"{v:.3e}", ha='center',
                            va='bottom' if v >= 0 else 'top', fontsize=8)
                
                # Annotate which feature was selected
                if abs(coef1) > 1e-6 and abs(coef2) < 1e-6:
                    ax.text(0, coef1, '✓ Selected', ha='center', va='bottom' if coef1 > 0 else 'top')
                elif abs(coef2) > 1e-6 and abs(coef1) < 1e-6:
                    ax.text(1, coef2, '✓ Selected', ha='center', va='bottom' if coef2 > 0 else 'top')
                elif abs(coef1) > 1e-6 and abs(coef2) > 1e-6:
                    ax.text(0.5, max(coef1, coef2), '✓ Both kept', ha='center', 
                           va='bottom', transform=ax.get_xaxis_transform())
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{save_path}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved multicollinearity analysis: {self.output_dir}/{save_path}")
        plt.close()


if __name__ == "__main__":
    print("Visualization module loaded successfully")
