import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from main import load_data
from main import center_data

def test():
    print("\n🧪 Testing center_data...")
    
    # Test on small known data
    X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mean_vec = np.mean(X_test, axis=0)
    B = center_data(X_test, mean_vec)
    
    assert B.shape == X_test.shape, f"Shape changed: {X_test.shape} -> {B.shape}"
    col_means = B.mean(axis=0)
    assert np.allclose(col_means, np.zeros(3), atol=1e-10), f"Column means not zero: {col_means}"
    
    # Test on actual digits
    X, _, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    col_means = B.mean(axis=0)
    assert np.allclose(col_means, np.zeros(64), atol=1e-10), "Digits centering failed"
    
    print("✅ center_data: All checks passed!")
    return True

if __name__ == "__main__":
    test()