"""
Adaptive LASSO Optimizer with Dynamic Soft-Thresholding
Implements proximal gradient descent with cooling schedule for λ
"""

import numpy as np
from tqdm import tqdm


class ProximalOperators:

    
    @staticmethod
    def soft_threshold(x, lambda_val):
       
        return np.sign(x) * np.maximum(np.abs(x) - lambda_val, 0)
    
    @staticmethod
    def compute_subdifferential_norm(beta, epsilon=1e-10):

        # For non-zero components, subdifferential is sign(beta)
        # For zero components, we use 0 (interior of subdifferential set)
        nonzero_mask = np.abs(beta) > epsilon
        subdiff = np.zeros_like(beta)
        subdiff[nonzero_mask] = np.sign(beta[nonzero_mask])
        
        return np.linalg.norm(subdiff)


class AdaptiveLassoOptimizer:

    
    def __init__(self, lambda_0=1.0, alpha=0.05, max_iter=1000,
                 tol=1e-6, learning_rate=0.01, use_lipschitz_step=False,
                 verbose=True):

        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.use_lipschitz_step = use_lipschitz_step
        self.verbose = verbose
        
        # Storage for tracking
        self.coef_ = None
        self.intercept_ = 0.0
        self.lambda_history_ = []
        self.coef_history_ = []
        self.loss_history_ = []
        self.n_iter_ = 0
        self.sparsity_history_ = []
        
    def _compute_loss(self, X, y, beta):
    
        n = X.shape[0]
        residuals = y - X @ beta
        mse_term = (0.5 / n) * np.sum(residuals ** 2)
        l1_term = self.lambda_history_[-1] * np.sum(np.abs(beta))
        return mse_term + l1_term
    
    def _update_lambda(self, beta, lambda_prev, iteration, n_features):
        
        subdiff_norm = ProximalOperators.compute_subdifferential_norm(beta)
        scaled_norm = subdiff_norm / max(np.sqrt(n_features), 1e-12)
        anneal = 1.0 / (iteration + 1)
        lambda_new = lambda_prev * np.exp(-self.alpha * scaled_norm * anneal)
        
        # Ensure λ doesn't become too small (numerical stability)
        lambda_new = max(lambda_new, 1e-6)
        
        return lambda_new
    
    def fit(self, X, y):
        
        n_samples, n_features = X.shape

        # Reset tracking (important when re-fitting the same instance)
        self.lambda_history_ = []
        self.coef_history_ = []
        self.loss_history_ = []
        self.sparsity_history_ = []
        self.n_iter_ = 0
        
        # Handle intercept by centering the target
        self.intercept_ = np.mean(y)
        y_centered = y - self.intercept_
        
        # Learning rate (optional Lipschitz step size)
        if self.use_lipschitz_step:
            spectral_norm = np.linalg.norm(X, ord=2)
            L = (spectral_norm ** 2) / n_samples
            lr = 1.0 / max(L, 1e-12)
        else:
            lr = self.learning_rate

        # Initialize coefficients
        beta = np.zeros(n_features)
        lambda_t = self.lambda_0
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("ADAPTIVE LASSO OPTIMIZATION")
            print("=" * 80)
            print(f"Initial λ: {self.lambda_0:.6f}")
            print(f"Cooling rate α: {self.alpha:.6f}")
            print(f"Learning rate: {lr:.6f}")
            print(f"Max iterations: {self.max_iter}")
            print("-" * 80)
        
        # Optimization loop
        iterator = tqdm(range(self.max_iter), disable=not self.verbose)
        for iteration in iterator:
            # Store current state
            self.lambda_history_.append(lambda_t)
            self.coef_history_.append(beta.copy())
            
            # Compute loss
            loss = self._compute_loss(X, y_centered, beta)
            self.loss_history_.append(loss)
            
            # Compute sparsity (percentage of zero coefficients)
            sparsity = 100 * np.mean(np.abs(beta) < 1e-6)
            self.sparsity_history_.append(sparsity)
            
            # Update progress bar
            if self.verbose:
                iterator.set_description(
                    f"Iter {iteration+1:4d} | Loss: {loss:.6f} | "
                    f"λ: {lambda_t:.6f} | Sparsity: {sparsity:.1f}%"
                )
            
            # Gradient of smooth part: -(1/n)X^T(y_centered - Xβ)
            residuals = y_centered - X @ beta
            gradient = -(1 / n_samples) * (X.T @ residuals)
            
            # Gradient descent step
            beta_tilde = beta - lr * gradient
            
            # Proximal operator (soft-thresholding)
            beta_new = ProximalOperators.soft_threshold(beta_tilde,
                                                        lr * lambda_t)
            
            # Check convergence
            coef_change = np.linalg.norm(beta_new - beta)
            if coef_change < self.tol:
                if self.verbose:
                    print(f"\n✓ Converged at iteration {iteration+1}")
                    print(f"  Coefficient change: {coef_change:.8f} < {self.tol}")
                self.n_iter_ = iteration + 1
                break
            
            # Update λ using cooling schedule
            lambda_t = self._update_lambda(beta_new, lambda_t, iteration, n_features)
            
            # Update coefficients
            beta = beta_new
        else:
            self.n_iter_ = self.max_iter
            if self.verbose:
                print(f"\n⚠ Maximum iterations reached ({self.max_iter})")
        
        # Store final model
        self.coef_ = beta
        self.lambda_history_.append(lambda_t)
        self.coef_history_.append(beta.copy())
        
        # Final statistics
        final_sparsity = 100 * np.mean(np.abs(beta) < 1e-6)
        non_zero_coefs = np.sum(np.abs(beta) >= 1e-6)
        
        if self.verbose:
            print("-" * 80)
            print("OPTIMIZATION COMPLETE")
            print("-" * 80)
            print(f"Final sparsity: {final_sparsity:.2f}%")
            print(f"Non-zero coefficients: {non_zero_coefs}/{n_features}")
            print(f"Final λ: {lambda_t:.6f}")
            print(f"Final loss: {self.loss_history_[-1]:.6f}")
            print("=" * 80 + "\n")
        
        return self
    
    def predict(self, X):
        
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return X @ self.coef_ + self.intercept_
    
    def get_feature_importance(self, feature_names=None):
       
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.coef_))]
        
        importance = [
            (name, coef, abs(coef)) 
            for name, coef in zip(feature_names, self.coef_)
        ]
        
        # Sort by absolute value (descending)
        importance.sort(key=lambda x: x[2], reverse=True)
        
        return importance


class StandardLasso:
    """Standard LASSO with fixed λ for comparison."""
    
    def __init__(self, lambda_val=0.1, max_iter=1000, learning_rate=0.01,
                 tol=1e-6, use_lipschitz_step=False, verbose=True):
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.use_lipschitz_step = use_lipschitz_step
        self.verbose = verbose
        
        self.coef_ = None
        self.intercept_ = 0.0
        self.loss_history_ = []
        self.coef_history_ = []
        self.sparsity_history_ = []
        self.n_iter_ = 0
    
    def fit(self, X, y):
        """Fit standard LASSO with fixed λ."""
        n_samples, n_features = X.shape

        # Reset tracking (important when re-fitting the same instance)
        self.loss_history_ = []
        self.coef_history_ = []
        self.sparsity_history_ = []
        self.n_iter_ = 0
        
        # Handle intercept by centering the target
        self.intercept_ = np.mean(y)
        y_centered = y - self.intercept_
        
        # Learning rate (optional Lipschitz step size)
        if self.use_lipschitz_step:
            spectral_norm = np.linalg.norm(X, ord=2)
            L = (spectral_norm ** 2) / n_samples
            lr = 1.0 / max(L, 1e-12)
        else:
            lr = self.learning_rate

        beta = np.zeros(n_features)
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("STANDARD LASSO OPTIMIZATION")
            print("=" * 80)
            print(f"Fixed λ: {self.lambda_val:.6f}")
            print(f"Learning rate: {lr:.6f}")
            print("-" * 80)
        
        iterator = tqdm(range(self.max_iter), disable=not self.verbose)
        for iteration in iterator:
            self.coef_history_.append(beta.copy())
            
            # Compute loss
            residuals = y_centered - X @ beta
            mse_term = (0.5 / n_samples) * np.sum(residuals ** 2)
            l1_term = self.lambda_val * np.sum(np.abs(beta))
            loss = mse_term + l1_term
            self.loss_history_.append(loss)
            
            sparsity = 100 * np.mean(np.abs(beta) < 1e-6)
            self.sparsity_history_.append(sparsity)
            
            if self.verbose:
                iterator.set_description(
                    f"Iter {iteration+1:4d} | Loss: {loss:.6f} | Sparsity: {sparsity:.1f}%"
                )
            
            # Gradient descent + proximal step
            gradient = -(1 / n_samples) * (X.T @ residuals)
            beta_tilde = beta - lr * gradient
            beta_new = ProximalOperators.soft_threshold(beta_tilde,
                                                        lr * self.lambda_val)
            
            # Check convergence
            if np.linalg.norm(beta_new - beta) < self.tol:
                self.n_iter_ = iteration + 1
                if self.verbose:
                    print(f"\n✓ Converged at iteration {iteration+1}")
                break
            
            beta = beta_new
        else:
            self.n_iter_ = self.max_iter
        
        self.coef_ = beta
        self.coef_history_.append(beta.copy())
        
        if self.verbose:
            final_sparsity = 100 * np.mean(np.abs(beta) < 1e-6)
            print(f"Final sparsity: {final_sparsity:.2f}%")
            print("=" * 80 + "\n")
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    # Simple test
    print("Testing Adaptive LASSO Optimizer...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:10] = np.random.randn(10) * 5  # Only 10 features are relevant
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    # Fit model
    model = AdaptiveLassoOptimizer(lambda_0=1.0, alpha=0.1, max_iter=500)
    model.fit(X, y)
    
    print(f"\n✓ True non-zero coefficients: 10")
    print(f"✓ Recovered non-zero coefficients: {np.sum(np.abs(model.coef_) >= 1e-6)}")
