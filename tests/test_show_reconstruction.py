import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from main import load_data
from main import center_data
from main import covariance_matrix
from main import eigen_decomposition
from main import show_sample_reconstruction

def test():
    print("\n🧪 Testing show_sample_reconstruction...")
    
    X, y, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    C = covariance_matrix(B)
    _, eigvecs = eigen_decomposition(C)
    
    for k in [2, 10, 30]:
        plt.close('all')
        try:
            show_sample_reconstruction(X, B, eigvecs, mean_vec, y, k=k, sample_index=0)
            plt.close('all')
        except Exception as e:
            assert False, f"Failed for k={k}: {e}"
    
    for idx in [0, 100, 500, 1000]:
        plt.close('all')
        try:
            show_sample_reconstruction(X, B, eigvecs, mean_vec, y, k=10, sample_index=idx)
            plt.close('all')
        except Exception as e:
            assert False, f"Failed for sample {idx}: {e}"
    
    print("✅ show_sample_reconstruction: All checks passed!")
    return True

if __name__ == "__main__":
    test()