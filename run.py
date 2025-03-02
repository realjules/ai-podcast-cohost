"""
AI Podcast Co-Host Runner Script

This script provides a convenient way to run the AI Podcast Co-Host application
from the root directory of the project, rather than from inside the app directory.
"""

import sys
import os
import subprocess
import webbrowser
from pathlib import Path

# Add the current directory to the path so we can import the app module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Create static directories
try:
    Path("app/static/uploads").mkdir(parents=True, exist_ok=True)
    Path("app/static/audio").mkdir(parents=True, exist_ok=True)
    # Create empty audio placeholder if it doesn't exist
    audio_placeholder = Path("app/static/audio/no_audio.mp3")
    if not audio_placeholder.exists():
        audio_placeholder.touch()
except Exception as e:
    print(f"Warning: Could not create static directories: {e}")

# Run the main module from the app package
if __name__ == "__main__":
    # First, check for the mock HTML file
    mock_html = Path("app/static/index.html")
    
    try:
        from app import main
        print("Starting AI Podcast Co-Host...")
        main.run()
    except ImportError as e:
        print(f"Error importing application: {e}")
        print("\nRunning the app directly...")
        # Fall back to running the module directly
        result = subprocess.run([sys.executable, "app/main.py"], capture_output=True, text=True)
        print(result.stdout)
        
        # If we're in mock mode and the HTML file exists, offer to open it
        if mock_html.exists():
            print("\nWould you like to open the mock interface in your browser? (y/n)")
            response = input().strip().lower()
            if response == 'y' or response == 'yes':
                file_url = mock_html.absolute().as_uri()
                print(f"Opening {file_url}")
                webbrowser.open(file_url)