"""
Run all test modules individually.
This will execute each test_*.py file one by one.
"""
import subprocess
import sys

def run_all_tests():
    print("\n" + "="*60)
    print("RUNNING ALL TESTS INDIVIDUALLY")
    print("="*60)
    
    test_files = [
        "test_load_data.py",
        "test_center_data.py",
        "test_covariance_matrix.py",
        "test_qr_algorithm.py",
        "test_demo_small_matrix.py",
        "test_eigen_decomposition.py",
        "test_explained_variance.py",
        "test_dimension_reduction.py",
        "test_visualize_2d.py",
        "test_reconstruction_error.py",
        "test_show_reconstruction.py",
        "test_m_less_than_n.py",
        "test_edge_cases.py"
    ]
    
    failed_tests = []
    
    for file in test_files:
        print(f"\n🔄 Executing {file}...")
        result = subprocess.run([sys.executable, file], capture_output=False, text=True)
        if result.returncode != 0:
            failed_tests.append(file)
            print(f"❌ {file} FAILED!")
        else:
            print(f"✅ {file} PASSED!")
    
    print("\n" + "="*60)
    if failed_tests:
        print(f"❌ Some tests failed: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        sys.exit(0)

if __name__ == "__main__":
    run_all_tests()