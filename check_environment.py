import sys

def check_python():
    print(f"Python version: {sys.version}")
    assert sys.version_info >= (3, 10), "Python 3.10+ is required."

def check_packages():
    import importlib
    required = [
        "streamlit",
        "transformers",
        "torch",
        "PIL",
        "huggingface_hub"
    ]
    for pkg in required:
        try:
            importlib.import_module(pkg)
            print(f"✅ {pkg} is installed.")
        except ImportError:
            print(f"❌ {pkg} is NOT installed.")
            raise

if __name__ == "__main__":
    check_python()
    check_packages()
    print("✅ Environment check passed!")