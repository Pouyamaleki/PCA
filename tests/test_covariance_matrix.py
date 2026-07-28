import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from main import load_data, center_data, covariance_matrix

def test():
    print("\n🧪 Testing covariance_matrix...")
    
    X_test = np.array([[1, 2], [3, 4], [5, 6]])
    mean_vec = np.mean(X_test, axis=0)
    B = center_data(X_test, mean_vec)
    C = covariance_matrix(B)
    
    assert C.shape == (2, 2), f"Expected (2,2), got {C.shape}"
    assert np.allclose(C, C.T), "Covariance matrix is not symmetric"
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals}"
    
    X, _, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    C = covariance_matrix(B)
    
    assert C.shape == (64, 64), f"Expected (64,64), got {C.shape}"
    assert np.allclose(C, C.T), "Digits covariance matrix is not symmetric"
    assert np.all(np.diag(C) >= 0), "Negative variance values"
    
    print("✅ covariance_matrix: All checks passed!")
    return True

if __name__ == "__main__":
    test()