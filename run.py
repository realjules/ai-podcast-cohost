"""
AI Podcast Co-Host Runner Script

This script provides a convenient way to run the AI Podcast Co-Host application
from the root directory of the project, rather than from inside the app directory.
"""

import sys
import os
import subprocess

# Add the current directory to the path so we can import the app module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Run the main module from the app package
if __name__ == "__main__":
    try:
        from app import main
        print("Starting AI Podcast Co-Host...")
        main.run()
    except ImportError as e:
        print(f"Error importing application: {e}")
        print("\nRunning the app directly...")
        # Fall back to running the module directly
        subprocess.run([sys.executable, "app/main.py"])