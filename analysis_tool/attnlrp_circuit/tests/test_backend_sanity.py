import sys
import os

# Add parent dir to path to allow importing backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend import ModelManager, AttributionEngine

def test_structure():
    print("Testing package structure...")
    try:
        mm = ModelManager()
        print("ModelManager class found.")
    except ImportError as e:
        print(f"Failed to import ModelManager: {e}")
        return

    try:
        ae = AttributionEngine(mm) 
        print("AttributionEngine class found.")
    except ImportError as e:
        print(f"Failed to import AttributionEngine: {e}")
        return
        
    print("Basic structure validity check passed.")

if __name__ == "__main__":
    test_structure()
