"""
Run all tests using pytest
"""
import pytest
import sys

if __name__ == "__main__":
    # run every test case from the tests folder
    sys.exit(pytest.main(["tests/", "-v", "--tb=short"]))