import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from main import load_data
from main import center_data
from main import covariance_matrix
from main import eigen_decomposition
from main import visualize_2d

def test():
    print("\n🧪 Testing visualize_2d...")
    
    X, y, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    C = covariance_matrix(B)
    _, eigvecs = eigen_decomposition(C)
    
    plt.close('all')
    T2 = visualize_2d(B, eigvecs, y)
    plt.close('all')
    
    assert T2.shape == (1797, 2), f"Expected T2 (1797,2), got {T2.shape}"
    W2_calc = eigvecs[:, :2]
    T2_calc = B @ W2_calc
    assert np.allclose(T2, T2_calc, atol=1e-10), "T2 calculation incorrect"
    assert np.allclose(W2_calc.T @ W2_calc, np.eye(2), atol=1e-10), "First 2 eigenvectors not orthonormal"
    
    print("✅ visualize_2d: All checks passed!")
    return True

if __name__ == "__main__":
    test()