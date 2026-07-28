import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from main import load_data
from main import center_data
from main import covariance_matrix
from main import eigen_decomposition

def test():
    print("\n🧪 Testing eigen_decomposition...")
    
    # Known matrix
    A = np.array([[4, 1], [1, 3]])
    expected = np.array([4.61803399, 2.38196601])
    eigvals, eigvecs = eigen_decomposition(A)
    
    assert np.all(eigvals[:-1] >= eigvals[1:]), "Eigenvalues not sorted descending"
    assert np.allclose(eigvals, np.sort(expected)[::-1], atol=1e-5), "Eigenvalues mismatch"
    assert np.allclose(eigvecs.T @ eigvecs, np.eye(2), atol=1e-10), "Eigenvectors not orthonormal"
    A_recon = eigvecs @ np.diag(eigvals) @ eigvecs.T
    assert np.allclose(A, A_recon, atol=1e-10), "Reconstruction failed"
    
    # Digits data
    X, _, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    C = covariance_matrix(B)
    eigvals, eigvecs = eigen_decomposition(C)
    
    assert len(eigvals) == 64, f"Expected 64 eigenvalues, got {len(eigvals)}"
    assert eigvecs.shape == (64, 64), f"Expected (64,64), got {eigvecs.shape}"
    assert np.allclose(eigvecs.T @ eigvecs, np.eye(64), atol=1e-10), "Eigenvectors not orthonormal"
    assert np.all(eigvals >= -1e-10), "Negative eigenvalues in covariance"
    
    print("✅ eigen_decomposition: All checks passed!")
    return True

if __name__ == "__main__":
    test()