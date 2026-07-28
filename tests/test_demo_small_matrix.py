import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from main import demo_on_small_matrix



def test():
    print("\n🧪 Testing demo_on_small_matrix...")
    
    A, Q, R, A1 = demo_on_small_matrix()
    
    assert isinstance(A, np.ndarray)
    assert isinstance(Q, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(A1, np.ndarray)
    
    assert A.shape == (4, 4)
    assert Q.shape == (4, 4)
    assert R.shape == (4, 4)
    assert A1.shape == (4, 4)
    
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-10), "Q is not orthogonal"
    eig_A = np.sort(np.linalg.eigvalsh(A))
    eig_A1 = np.sort(np.linalg.eigvalsh(A1))
    assert np.allclose(eig_A, eig_A1, atol=1e-10), "A and A1 do not have same eigenvalues"
    
    rank = np.linalg.matrix_rank(A)
    assert rank <= 4, "Rank should be <= 4"
    assert np.linalg.matrix_rank(Q) == Q.shape[1], "Columns of Q are not linearly independent"
    
    print("✅ demo_on_small_matrix: All checks passed!")
    return True

if __name__ == "__main__":
    test()