import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from main import load_data, center_data, covariance_matrix, eigen_decomposition, explained_variance

def test():
    print("\n🧪 Testing explained_variance...")
    
    eigvals = np.array([10, 5, 3, 2])
    explained, cum, k_90 = explained_variance(eigvals)
    plt.close('all')
    
    assert np.isclose(np.sum(explained), 1.0, atol=1e-10), f"Sum should be 1, got {np.sum(explained)}"
    assert np.all(cum[1:] >= cum[:-1]), "Cumulative variance is not increasing"
    assert k_90 == 3, f"Expected k_90=3, got {k_90}"
    
    # Digits data
    X, _, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    C = covariance_matrix(B)
    eigvals, _ = eigen_decomposition(C)
    explained, cum, k_90 = explained_variance(eigvals)
    plt.close('all')
    
    assert np.isclose(np.sum(explained), 1.0, atol=1e-10), "Ratios don't sum to 1"
    assert np.isclose(cum[-1], 1.0, atol=1e-10), "Cumulative doesn't end at 1"
    assert 1 <= k_90 <= 64, f"k_90={k_90} out of bounds"
    assert cum[k_90 - 1] >= 0.90, f"With k={k_90}, variance = {cum[k_90-1]:.4f} < 0.90"
    
    print("✅ explained_variance: All checks passed!")
    return True

if __name__ == "__main__":
    test()