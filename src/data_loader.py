"""
Data Loader for House Prices Dataset
Handles preprocessing, missing values, one-hot encoding, and correlation analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class HousePriceDataLoader:
    """
    Comprehensive data loader for House Prices dataset with feature engineering
    and multicollinearity detection.
    """
    
    def __init__(self, filepath='data/train.csv', test_size=0.2, random_state=42):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        self.filepath = filepath
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.correlated_pairs = []
        
    def load_and_preprocess(self):
        """
        Main pipeline for loading and preprocessing data.
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Preprocessed training and testing data
        """
        print("=" * 80)
        print("DATA LOADING AND PREPROCESSING")
        print("=" * 80)
        
        # Load data
        df = pd.read_csv(self.filepath)
        print(f"\n✓ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Separate target variable
        y = df['SalePrice'].values
        X = df.drop(['Id', 'SalePrice'], axis=1)
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        X = self._encode_categorical(X)
        
        # Store feature names before splitting
        self.feature_names = X.columns.tolist()
        
        # Analyze correlations
        self._analyze_correlations(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Log-transform target to reduce skewness
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
        
        print(f"\n✓ Final dataset shape: X_train={X_train.shape}, X_test={X_test.shape}")
        print(f"✓ Total features after preprocessing: {X_train.shape[1]}")
        print("=" * 80)
        
        return X_train, X_test, y_train, y_test
    
    def _handle_missing_values(self, X):
        """
        Handle missing values using intelligent imputation.
        
        Parameters:
        -----------
        X : DataFrame
            Input features
            
        Returns:
        --------
        X : DataFrame
            Features with imputed values
        """
        print("\n[Step 1] Handling Missing Values")
        print("-" * 40)
        
        missing_counts = X.isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]
        
        if len(features_with_missing) > 0:
            print(f"  Features with missing values: {len(features_with_missing)}")
            
            # Numerical features: fill with median
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if X[col].isnull().sum() > 0:
                    X[col] = X[col].fillna(X[col].median())
            
            # Categorical features: fill with mode or 'None'
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if X[col].isnull().sum() > 0:
                    fill_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'None'
                    X[col] = X[col].fillna(fill_val)
            
            print(f"  ✓ Imputed missing values successfully")
        else:
            print(f"  ✓ No missing values found")
        
        return X
    
    def _encode_categorical(self, X):
        """
        One-hot encode categorical variables.
        
        Parameters:
        -----------
        X : DataFrame
            Input features
            
        Returns:
        --------
        X : DataFrame
            Encoded features
        """
        print("\n[Step 2] Encoding Categorical Variables")
        print("-" * 40)
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"  Categorical features found: {len(categorical_cols)}")
        
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            print(f"  ✓ One-hot encoding completed")
        
        return X
    
    def _analyze_correlations(self, X, threshold=0.8):
        """
        Identify highly correlated feature pairs for multicollinearity analysis.
        
        Parameters:
        -----------
        X : DataFrame
            Input features
        threshold : float
            Correlation threshold (default: 0.8)
        """
        print("\n[Step 3] Multicollinearity Analysis")
        print("-" * 40)
        
        # Compute correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle (avoid duplicates)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs
        self.correlated_pairs = []
        for column in upper_triangle.columns:
            correlated_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()
            for feature in correlated_features:
                self.correlated_pairs.append({
                    'feature_1': column,
                    'feature_2': feature,
                    'correlation': corr_matrix.loc[column, feature]
                })
        
        if len(self.correlated_pairs) > 0:
            print(f"  Highly correlated pairs (ρ > {threshold}): {len(self.correlated_pairs)}")
            print("\n  Top 5 correlated pairs:")
            for i, pair in enumerate(sorted(self.correlated_pairs, 
                                           key=lambda x: x['correlation'], 
                                           reverse=True)[:5]):
                print(f"    {i+1}. {pair['feature_1']} ↔ {pair['feature_2']}: ρ = {pair['correlation']:.3f}")
        else:
            print(f"  ✓ No highly correlated pairs found (threshold: {threshold})")
    
    def get_feature_names(self):
        """Return the list of feature names."""
        return self.feature_names
    
    def get_correlated_pairs(self):
        """Return the list of highly correlated feature pairs."""
        return self.correlated_pairs
    
    def inverse_transform_target(self, y):
        """
        Inverse log transformation for predictions.
        
        Parameters:
        -----------
        y : array
            Log-transformed target values
            
        Returns:
        --------
        array : Original scale prices
        """
        return np.expm1(y)


if __name__ == "__main__":
    # Test the data loader
    loader = HousePriceDataLoader()
    X_train, X_test, y_train, y_test = loader.load_and_preprocess()
    
    print("\n" + "=" * 80)
    print("SAMPLE STATISTICS")
    print("=" * 80)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Target mean (log-scale): {y_train.mean():.4f}")
    print(f"Target std (log-scale): {y_train.std():.4f}")
    print("=" * 80)
