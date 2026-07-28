import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from main import load_data
from main import m_less_than_n

def test():
    print("\n🧪 Testing m_less_than_n...")
    
    X, y, _ = load_data()
    
    plt.close('all')
    eigvals_subset, eigvecs_subset, X_centered_subset = m_less_than_n(X, y)
    plt.close('all')
    
    m, n = 50, 64
    assert eigvals_subset.shape == (n,), f"Expected eigenvalues shape ({n},), got {eigvals_subset.shape}"
    assert eigvecs_subset.shape == (n, n), f"Expected eigenvectors shape ({n},{n}), got {eigvecs_subset.shape}"
    assert X_centered_subset.shape == (m, n), f"Expected centered subset shape ({m},{n}), got {X_centered_subset.shape}"
    assert np.allclose(eigvecs_subset.T @ eigvecs_subset, np.eye(n), atol=1e-10), "Eigenvectors not orthonormal"
    
    tolerance = 1e-10
    non_zero_count = np.sum(eigvals_subset >= tolerance)
    assert non_zero_count <= m - 1, f"Non-zero eigenvalues {non_zero_count} should be <= {m-1}"
    
    print("✅ m_less_than_n: All checks passed!")
    return True

if __name__ == "__main__":
    test()