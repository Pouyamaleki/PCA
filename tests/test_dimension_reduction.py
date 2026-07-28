import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from main import load_data, center_data, covariance_matrix, eigen_decomposition, dimension_reduction

def test():
    print("\n🧪 Testing dimension_reduction...")
    
    X_test = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    mean_vec = np.mean(X_test, axis=0)
    B = center_data(X_test, mean_vec)
    C = covariance_matrix(B)
    _, eigvecs = eigen_decomposition(C)
    
    k = 2
    W, T = dimension_reduction(B, eigvecs, k)
    assert W.shape == (4, k), f"Expected W (4,{k}), got {W.shape}"
    assert T.shape == (3, k), f"Expected T (3,{k}), got {T.shape}"
    assert np.allclose(W.T @ W, np.eye(k), atol=1e-10), "W columns are not orthonormal"
    assert np.allclose(T, B @ W, atol=1e-10), "T calculation incorrect"
    
    # Digits data
    X, _, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    C = covariance_matrix(B)
    _, eigvecs = eigen_decomposition(C)
    
    for k in [1, 10, 30, 50]:
        W, T = dimension_reduction(B, eigvecs, k)
        assert W.shape == (64, k), f"W shape mismatch for k={k}"
        assert T.shape == (1797, k), f"T shape mismatch for k={k}"
        assert np.allclose(W.T @ W, np.eye(k), atol=1e-10), f"W not orthonormal for k={k}"
    
    print("✅ dimension_reduction: All checks passed!")
    return True

if __name__ == "__main__":
    test()