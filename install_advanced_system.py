#!/usr/bin/env python3
"""
Installation Script for Advanced Character Processing System
Installs required dependencies and sets up the system
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"Python {version.major}.{version.minor} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Core dependencies
    core_deps = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "tqdm>=4.64.0",
        "colorama>=0.4.4"
    ]
    
    # Gender classification libraries
    gender_deps = [
        "gender-guesser>=0.4.0",
        "chicksexer>=0.1.0"
    ]
    
    # Text processing libraries
    text_deps = [
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.12.0"
    ]
    
    # Personality analysis models
    personality_deps = [
        "datasets>=2.0.0"
    ]
    
    all_deps = core_deps + gender_deps + text_deps + personality_deps
    
    success = True
    for dep in all_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            success = False
    
    return success

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        "characters",
        "dialogue", 
        "audio",
        "batch_results",
        "modelsbooknlp/output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def test_installation():
    """Test if the installation works"""
    print("\nTesting installation...")
    
    try:
        # Test imports
        import pandas
        import numpy
        import transformers
        import torch
        import gender_guesser
        from fuzzywuzzy import fuzz
        print("All core dependencies imported successfully")
        
        # Test advanced character processor
        from advanced_character_processor import AdvancedCharacterProcessor
        processor = AdvancedCharacterProcessor()
        print("AdvancedCharacterProcessor imported and initialized")
        
        return True
        
    except ImportError as e:
        print(f"Import test failed: {e}")
        return False
    except Exception as e:
        print(f"Initialization test failed: {e}")
        return False

def main():
    """Main installation function"""
    print("Advanced Character Processing System - Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\nInstallation failed. Please check the errors above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("\nInstallation test failed. Please check the errors above.")
        sys.exit(1)
    
    print("\nInstallation completed successfully!")
    print("\nNext steps:")
    print("1. Ensure BookNLP is installed and configured")
    print("2. Place your BookNLP output files in modelsbooknlp/output/")
    print("3. Run: python advanced_character_processor.py")
    print("4. Or run: python run_advanced_pipeline.py --book-ids samples --report")
    print("5. For demonstration: python advanced_example_usage.py")
    
    print("\nDocumentation:")
    print("- Read ADVANCED_SYSTEM_README.md for detailed usage")
    print("- Check requirements.txt for all dependencies")
    print("- Review config.json for configuration options")

if __name__ == "__main__":
    main()
