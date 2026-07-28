import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from main import center_data
from main import covariance_matrix
from main import eigen_decomposition
from main import dimension_reduction

def test():
    print("\n🧪 Testing edge_cases...")
    
    # 1D data
    X_1d = np.array([[1], [2], [3], [4]])
    mean_vec = np.mean(X_1d, axis=0)
    B = center_data(X_1d, mean_vec)
    C = covariance_matrix(B)
    assert C.shape == (1, 1), f"Expected (1,1), got {C.shape}"
    assert np.allclose(C, C.T), "1D covariance not symmetric"
    
    # Constant data
    X_const = np.array([[5, 5], [5, 5], [5, 5]])
    mean_vec = np.mean(X_const, axis=0)
    B = center_data(X_const, mean_vec)
    C = covariance_matrix(B)
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(np.abs(eigvals) < 1e-10), f"Constant data should have zero eigenvalues, got {eigvals}"
    
    # Extreme k
    X_test = np.random.randn(10, 5)
    mean_vec = np.mean(X_test, axis=0)
    B = center_data(X_test, mean_vec)
    C = covariance_matrix(B)
    _, eigvecs = eigen_decomposition(C)
    
    # k = 0
    try:
        W, T = dimension_reduction(B, eigvecs, 0)
        assert W.shape == (5, 0), f"Expected (5,0), got {W.shape}"
        assert T.shape == (10, 0), f"Expected (10,0), got {T.shape}"
    except Exception as e:
        assert False, f"dimension_reduction with k=0 failed: {e}"
    
    # k = n
    W, T = dimension_reduction(B, eigvecs, 5)
    assert W.shape == (5, 5), f"Expected (5,5), got {W.shape}"
    assert T.shape == (10, 5), f"Expected (10,5), got {T.shape}"
    assert np.allclose(W.T @ W, np.eye(5), atol=1e-10), "W not orthonormal for k=n"
    
    print("✅ edge_cases: All checks passed!")
    return True

if __name__ == "__main__":
    test()