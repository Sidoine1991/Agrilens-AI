"""
AgriLens AI - Plant Disease Diagnosis Application
Main entry point for Hugging Face Spaces deployment
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main application
from streamlit_app_multilingual import *

if __name__ == "__main__":
    # This file serves as the entry point for Hugging Face Spaces
    # The actual application logic is in src/streamlit_app_multilingual.py
    pass 