"""
Test script to verify the UPR Defense implementation.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.main import run_smoke_tests

if __name__ == "__main__":
    run_smoke_tests()
