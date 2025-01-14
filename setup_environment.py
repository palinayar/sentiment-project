import os
import subprocess
import sys

def install_requirements():
    """Install required libraries from requirements.txt."""
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("Error: requirements.txt file not found!")
            sys.exit(1)
        
        # Install dependencies
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
