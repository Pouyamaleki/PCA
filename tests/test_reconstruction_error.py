import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from main import load_data
from main import center_data
from main import covariance_matrix
from main import eigen_decomposition
from main import reconstruction_error

def test():
    print("\n🧪 Testing reconstruction_error...")
    
    X, y, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    C = covariance_matrix(B)
    _, eigvecs = eigen_decomposition(C)
    
    plt.close('all')
    errors = reconstruction_error(X, B, eigvecs, mean_vec, y)
    plt.close('all')
    
    expected_k = [1, 5, 10, 15, 20, 30, 40, 50, 64]
    assert len(errors) == len(expected_k), f"Expected {len(expected_k)} errors, got {len(errors)}"
    assert all(e >= 0 for e in errors), "MSE should be non-negative"
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i+1], f"Errors should be decreasing: {errors[i]} >= {errors[i+1]}"
    assert errors[-1] < 1e-10, f"Reconstruction with k=64 should be near zero, got {errors[-1]}"
    
    print("✅ reconstruction_error: All checks passed!")
    return True

if __name__ == "__main__":
    test()