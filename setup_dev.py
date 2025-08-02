#!/usr/bin/env python3
"""
AgriLens AI Development Setup Script
Automates the installation and configuration process for developers.
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path

def print_banner():
    """Print the application banner."""
    print("🌱" * 50)
    print("🌱 AgriLens AI - Development Setup")
    print("🌱" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Error: Python 3.11+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def check_system_requirements():
    """Check system requirements."""
    print("\n🔍 Checking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"   RAM: {memory_gb:.1f} GB")
        if memory_gb < 8:
            print("   ⚠️  Warning: Less than 8GB RAM detected")
        else:
            print("   ✅ Sufficient RAM detected")
    except ImportError:
        print("   ⚠️  psutil not available, cannot check memory")
    
    # Check disk space
    try:
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        print(f"   Disk space: {disk_gb:.1f} GB available")
        if disk_gb < 15:
            print("   ⚠️  Warning: Less than 15GB free space")
        else:
            print("   ✅ Sufficient disk space")
    except:
        print("   ⚠️  Cannot check disk space")

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print("   ✅ Pip upgraded")
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("   ✅ Dependencies installed")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error installing dependencies: {e}")
        return False
    
    return True

def setup_environment():
    """Setup environment variables and directories."""
    print("\n⚙️  Setting up environment...")
    
    # Create necessary directories
    directories = ["logs", "models", "exports", "cache"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/ directory")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# AgriLens AI Environment Variables
HF_TOKEN=your_huggingface_token_here
MODEL_CACHE_DIR=./models
LOG_LEVEL=INFO
DEBUG_MODE=false
"""
        env_file.write_text(env_content)
        print("   ✅ Created .env file")
        print("   ⚠️  Please update HF_TOKEN in .env file")

def check_gpu():
    """Check for GPU availability."""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   ✅ GPU detected: {gpu_name}")
            print(f"   ✅ GPU memory: {gpu_memory:.1f} GB")
        else:
            print("   ℹ️  No CUDA GPU detected, will use CPU")
    except ImportError:
        print("   ⚠️  PyTorch not available, cannot check GPU")

def validate_config():
    """Validate configuration files."""
    print("\n🔧 Validating configuration...")
    
    # Check config.json
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        print("   ✅ config.json is valid")
    except Exception as e:
        print(f"   ❌ Error in config.json: {e}")
        return False
    
    # Check requirements.txt
    if Path("requirements.txt").exists():
        print("   ✅ requirements.txt exists")
    else:
        print("   ❌ requirements.txt not found")
        return False
    
    return True

def run_tests():
    """Run basic tests."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        import streamlit
        import torch
        import transformers
        print("   ✅ Core imports successful")
        
        # Test configuration loading
        with open("config.json", "r") as f:
            config = json.load(f)
        print("   ✅ Configuration loading successful")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False
    
    return True

def print_next_steps():
    """Print next steps for the developer."""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Update HF_TOKEN in .env file with your Hugging Face token")
    print("   2. Run the application: streamlit run src/streamlit_app_multilingual.py")
    print("   3. Access the app at: http://localhost:8501")
    print("\n📚 Useful commands:")
    print("   - Start app: streamlit run src/streamlit_app_multilingual.py")
    print("   - Run tests: pytest")
    print("   - Format code: black .")
    print("   - Lint code: flake8")
    print("\n🔗 Resources:")
    print("   - Documentation: TECHNICAL_NOTE.md")
    print("   - Live demo: https://huggingface.co/spaces/sido1991/Agrilens_IAv1")
    print("   - GitHub: https://github.com/Sidoine1991/Agrilens-AI")

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Check system requirements
    check_system_requirements()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Check GPU
    check_gpu()
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("❌ Tests failed")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 