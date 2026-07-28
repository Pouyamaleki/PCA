import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from main import load_data

def test():
    print("\n🧪 Testing load_data...")
    
    X, y, mean_vec = load_data()
    
    assert isinstance(X, np.ndarray), "X should be numpy array"
    assert isinstance(y, np.ndarray), "y should be numpy array"
    assert isinstance(mean_vec, np.ndarray), "mean_vec should be numpy array"
    
    assert X.shape == (1797, 64), f"Expected (1797,64), got {X.shape}"
    assert y.shape == (1797,), f"Expected (1797,), got {y.shape}"
    assert mean_vec.shape == (64,), f"Expected (64,), got {mean_vec.shape}"
    
    assert np.all(X >= 0) and np.all(X <= 16), "Pixel values should be in [0, 16]"
    assert np.all(y >= 0) and np.all(y <= 9), "Labels should be digits 0-9"
    assert np.all(mean_vec >= 0) and np.all(mean_vec <= 16), "Mean values out of range"
    
    print("✅ load_data: All checks passed!")
    return True

if __name__ == "__main__":
    test()