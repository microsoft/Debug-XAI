import sys
import os

# Add parent dir to path to allow importing backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app import app, LoadModelRequest, ComputeLogitsRequest

def test_api_structure():
    print("Testing API structure...")
    try:
        req = LoadModelRequest()
        print("LoadModelRequest instantiated.")
        req2 = ComputeLogitsRequest(prompt="Test")
        print("ComputeLogitsRequest instantiated.")
        print("API structure validity check passed.")
    except Exception as e:
        print(f"API Check Failed: {e}")

if __name__ == "__main__":
    test_api_structure()
