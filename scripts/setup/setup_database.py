#!/usr/bin/env python3
"""
Setup script for EchoTales Enhanced system
Since we're using MongoDB with NextJS backend, this script validates system components
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.echotales.utils.logger_setup import setup_logger

logger = setup_logger(__name__, log_file="logs/setup.log")

def validate_directory_structure():
    """Validate that all required directories exist"""
    logger.info("Validating directory structure...")
    
    required_dirs = [
        "src/echotales/core",
        "src/echotales/processing", 
        "src/echotales/voice",
        "src/echotales/api",
        "src/echotales/utils",
        "models/booknlp",-
        "models/tts",
        "models/voice_cloning",
        "data/raw/books",
        "data/processed",
        "data/output",
        "data/voice_dataset",
        "configs",
        "logs",
        "cache"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
            logger.warning(f"Missing directory: {dir_path}")
    
    if missing_dirs:
        logger.error(f"Missing {len(missing_dirs)} required directories")
        return False
    
    logger.info("Directory structure validation passed")
    return True


def validate_configuration_files():
    """Validate configuration files exist and are readable"""
    logger.info("Validating configuration files...")
    
    config_files = [
        "configs/development.yaml",
        "requirements/base.txt",
        "requirements/development.txt"
    ]
    
    missing_configs = []
    for config_file in config_files:
        config_path = project_root / config_file
        if not config_path.exists():
            missing_configs.append(config_file)
            logger.warning(f"Missing config file: {config_file}")
    
    if missing_configs:
        logger.error(f"Missing {len(missing_configs)} configuration files")
        return False
    
    # Test loading development config
    try:
        from src.echotales.utils.config_loader import ConfigLoader
        config = ConfigLoader()
        logger.info(f"Configuration loaded successfully: {config.app.name} v{config.app.version}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False
    
    logger.info("Configuration validation passed")
    return True


def validate_python_environment():
    """Validate Python environment and dependencies"""
    logger.info("Validating Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 10:
        logger.error(f"Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    logger.info(f"Python version OK: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check critical imports
    critical_imports = [
        ('flask', 'Flask web framework'),
        ('torch', 'PyTorch ML framework'),
        ('transformers', 'HuggingFace transformers'),
        ('librosa', 'Audio processing'),
        ('sklearn', 'Scikit-learn ML'),
        ('spacy', 'SpaCy NLP'),
        ('redis', 'Redis client'),
        ('pydantic', 'Pydantic validation'),
        ('yaml', 'YAML configuration')
    ]
    
    import_failures = []
    for module_name, description in critical_imports:
        try:
            __import__(module_name)
            logger.info(f"✓ {module_name} ({description})")
        except ImportError as e:
            import_failures.append((module_name, description, str(e)))
            logger.error(f"✗ {module_name} ({description}): {e}")
    
    if import_failures:
        logger.error(f"Failed to import {len(import_failures)} critical modules")
        logger.error("Please install dependencies using: pip install -r requirements/development.txt")
        return False
    
    logger.info("Python environment validation passed")
    return True


def test_system_components():
    """Test that system components can be initialized"""
    logger.info("Testing system components...")
    
    try:
        # Test character processor
        from src.echotales.processing.enhanced_character_processor import EnhancedCharacterProcessor
        char_processor = EnhancedCharacterProcessor()
        logger.info("✓ Enhanced Character Processor initialized")
    except Exception as e:
        logger.error(f"✗ Character Processor failed: {e}")
        return False
    
    try:
        # Test scene analyzer
        from src.echotales.processing.scene_analyzer import SceneAnalyzer
        scene_analyzer = SceneAnalyzer()
        logger.info("✓ Scene Analyzer initialized")
    except Exception as e:
        logger.error(f"✗ Scene Analyzer failed: {e}")
        return False
    
    try:
        # Test voice classifier (may fail if models not downloaded)
        from src.echotales.voice.classifier import VoiceDatasetClassifier
        voice_classifier = VoiceDatasetClassifier(
            dataset_path="data/voice_dataset/",
            models_path="models/voice_cloning/"
        )
        logger.info("✓ Voice Dataset Classifier initialized")
    except Exception as e:
        logger.warning(f"⚠ Voice Classifier initialization warning: {e}")
        # Not critical failure - voice models may not be downloaded yet
    
    logger.info("System components test passed")
    return True


def create_sample_voice_dataset_structure():
    """Create sample structure for voice dataset"""
    logger.info("Creating voice dataset structure...")
    
    base_path = project_root / "data/voice_dataset"
    
    # Create male actor directories
    male_path = base_path / "male_actors"
    for i in range(1, 13):  # 12 male actors
        actor_dir = male_path / f"male_actor_{i:02d}"
        actor_dir.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder README
        readme_file = actor_dir / "README.md"
        if not readme_file.exists():
            with open(readme_file, 'w') as f:
                f.write(f"# Male Actor {i:02d}\n\n")
                f.write("Place 44 audio samples (.wav, .mp3, or .flac) in this directory.\n")
                f.write("Sample naming: sample_001.wav, sample_002.wav, etc.\n")
    
    # Create female actor directories
    female_path = base_path / "female_actors"
    for i in range(1, 13):  # 12 female actors
        actor_dir = female_path / f"female_actor_{i:02d}"
        actor_dir.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder README
        readme_file = actor_dir / "README.md"
        if not readme_file.exists():
            with open(readme_file, 'w') as f:
                f.write(f"# Female Actor {i:02d}\n\n")
                f.write("Place 44 audio samples (.wav, .mp3, or .flac) in this directory.\n")
                f.write("Sample naming: sample_001.wav, sample_002.wav, etc.\n")
    
    # Create processed samples directory
    processed_path = base_path / "processed_samples"
    processed_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Voice dataset structure created successfully")
    return True


def create_environment_activation_scripts():
    """Create activation scripts for different environments"""
    logger.info("Creating environment activation scripts...")
    
    environments = ['development', 'production', 'testing']
    
    for env in environments:
        script_content_bat = f"""@echo off
echo Activating EchoTales {env.title()} Environment...

call environments\\{env}\\Scripts\\activate.bat
echo Environment activated. Python path:
python -c "import sys; print(sys.executable)"

echo.
echo Available commands:
echo   python src/echotales/api/app.py    - Start Flask API server
echo   python scripts/utils/validate_setup.py  - Validate system setup
echo.
echo Environment: {env.upper()}
"""
        
        script_path = project_root / f"activate_{env}.bat"
        with open(script_path, 'w') as f:
            f.write(script_content_bat)
        
        logger.info(f"Created activation script: activate_{env}.bat")
    
    return True


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("EchoTales Enhanced System Setup")
    logger.info("=" * 60)
    
    all_tests_passed = True
    
    # Run validation tests
    tests = [
        ("Directory Structure", validate_directory_structure),
        ("Configuration Files", validate_configuration_files),
        ("Python Environment", validate_python_environment),
        ("System Components", test_system_components)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if not test_func():
                all_tests_passed = False
                logger.error(f"{test_name} validation failed")
            else:
                logger.info(f"{test_name} validation passed")
        except Exception as e:
            logger.error(f"{test_name} validation error: {e}")
            all_tests_passed = False
    
    # Setup tasks
    setup_tasks = [
        ("Voice Dataset Structure", create_sample_voice_dataset_structure),
        ("Environment Activation Scripts", create_environment_activation_scripts)
    ]
    
    for task_name, task_func in setup_tasks:
        logger.info(f"\n--- {task_name} ---")
        try:
            if not task_func():
                logger.warning(f"{task_name} setup failed")
            else:
                logger.info(f"{task_name} setup completed")
        except Exception as e:
            logger.error(f"{task_name} setup error: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    if all_tests_passed:
        logger.info("🎉 SETUP SUCCESSFUL! All validations passed.")
        logger.info("\nNext steps:")
        logger.info("1. Install dependencies: pip install -r requirements/development.txt")
        logger.info("2. Add voice samples to data/voice_dataset/ directories")
        logger.info("3. Set environment variables (GEMINI_API_KEY, etc.)")
        logger.info("4. Start the API server: python src/echotales/api/app.py")
    else:
        logger.error("❌ SETUP INCOMPLETE! Some validations failed.")
        logger.error("Please check the errors above and fix them before proceeding.")
    
    logger.info("=" * 60)
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())