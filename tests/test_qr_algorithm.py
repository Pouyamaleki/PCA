import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from main import qr_algorithm, load_data, center_data, covariance_matrix

def test():
    print("\n🧪 Testing qr_algorithm...")
    
    np.random.seed(42)
    
    for size in [3, 4, 5]:
        A = np.random.randn(size, size)
        A = A @ A.T
        Q_list = qr_algorithm(A, num_iterations=3)
        
        assert len(Q_list) == 3, f"Expected 3 Q matrices, got {len(Q_list)}"
        for i, Q in enumerate(Q_list):
            assert Q.shape == (size, size), f"Q{i} shape mismatch"
            assert np.allclose(Q.T @ Q, np.eye(size), atol=1e-10), f"Q{i} is not orthogonal"
            det = np.linalg.det(Q)
            assert np.isclose(abs(det), 1.0, atol=1e-10), f"Q{i} determinant should be ±1, got {det}"
    
    # Test with covariance matrix
    X, _, _ = load_data()
    mean_vec = np.mean(X, axis=0)
    B = center_data(X, mean_vec)
    C = covariance_matrix(B)
    Q_list = qr_algorithm(C, num_iterations=2)
    assert len(Q_list) == 2, "QR on digits data failed"
    
    print("✅ qr_algorithm: All checks passed!")
    return True

if __name__ == "__main__":
    test()