# EchoTales Enhanced - File Organization

This document describes the reorganized file structure for better project management.

## 📁 New Directory Structure

### `/setup/` - Setup and Installation Scripts
- `setup_models.py` - Simplified model initialization with HuggingFace integration
- `setup_all_models.py` - Comprehensive model setup (BookNLP, OpenVoice, Character Classification)  
- `quick_setup.bat` - Windows batch quick setup script
- `setup_windows.ps1` - Comprehensive Windows PowerShell setup script

**Usage:** Run these scripts to set up your EchoTales environment and download required models.

### `/processing/` - Data Processing Scripts
- `generate_json_files.py` - Comprehensive JSON data generation from preprocessed BookNLP data
- `process_real_booknlp_data.py` - Real BookNLP data processor (NO template/fake data)
- `process_chapter_with_samples.py` - Enhanced chapter processor with character voice samples
- `run_booknlp_from_start.py` - BookNLP setup and processing script from scratch

**Usage:** These scripts handle text processing, character extraction, and data preparation.

### `/audio_generation/` - Audio Creation Pipeline
- `voice_audio_generator.py` - Complete voice audio generation from JSON data using Edge TTS
- `create_chapter_audio_fixed.py` - Fixed version with MP3-to-WAV conversion using ffmpeg

**Usage:** These scripts generate audio files from processed text data using various TTS engines.

### `/utilities/` - Utility Scripts
- `classify_voice_data.py` - Voice dataset classification with robust error handling
- `quick_voice_samples.py` - Quick character voice sample generator

**Usage:** Helper scripts for data analysis and classification tasks.

### `/documentation/` - Project Documentation
- `project_completion_report.md` - Complete project summary and achievements
- `BOOKNLP_SETUP_COMPLETE.md` - BookNLP setup documentation and usage guide

**Usage:** Reference documentation for understanding project capabilities and setup procedures.

## 🗂️ Root Directory Files (Unchanged)
- `run_dev.py` - Development server launcher (kept in root for easy access)
- `.env.template` - Environment configuration template
- `.gitignore` - Git ignore patterns
- `requirements.txt` - Python dependencies
- `requirements-minimal.txt` - Minimal dependencies for quick start

## 🔄 Changes Made

### Files Moved:
- **Setup scripts** → `/setup/`
- **Processing scripts** → `/processing/`
- **Audio generation scripts** → `/audio_generation/`
- **Utility scripts** → `/utilities/`
- **Documentation files** → `/documentation/`

### Files Removed (Duplicates/Redundant):
- `create_chapter_audio.py` - Removed redundant version (kept the "fixed" version)
- `run_complete_audio_pipeline.py` - Removed (dependent on missing src modules)
- `run_line_by_line_processing.py` - Removed (dependent on missing src modules)  
- `run_voice_processing.py` - Removed (dependent on missing src modules)
- `booknlp/process_novel.py` - Removed (outdated, replaced by better processing scripts)
- `booknlp/test_booknlp.py` - Removed (outdated)
- `run_booknlp.py` - Removed (replaced by `processing/run_booknlp_from_start.py`)

### Files Preserved in Root:
- `run_dev.py` - Main development server (frequently used)
- Configuration files (`.env.template`, `.gitignore`)
- Requirements files
- Core project directories (`src/`, `data/`, etc.)

## 🚀 Quick Start Commands

```bash
# Setup (Windows)
.\setup\setup_windows.ps1

# Quick setup (Windows)
.\setup\quick_setup.bat

# Setup models
python .\setup\setup_models.py

# Run development server
python run_dev.py

# Process data from scratch
python .\processing\run_booknlp_from_start.py

# Process existing BookNLP data
python .\processing\generate_json_files.py

# Generate audio
python .\audio_generation\voice_audio_generator.py

# Quick voice samples
python .\utilities\quick_voice_samples.py
```

## 🎯 Core BookNLP → Characters.json Workflow

The cleaned project now focuses on the essential workflow:

1. **Setup:** `setup/setup_all_models.py` or `setup/setup_models.py`
2. **BookNLP Processing:** `processing/run_booknlp_from_start.py` 
3. **Data Processing:** `processing/process_real_booknlp_data.py`
4. **JSON Generation:** `processing/generate_json_files.py`
5. **Audio Generation:** `audio_generation/voice_audio_generator.py`
6. **Utilities:** `utilities/quick_voice_samples.py` for testing

## 📋 Benefits of This Organization

1. **Cleaner Structure** - Removed 7 redundant/duplicate files
2. **Focus on Core Functionality** - BookNLP → characters.json processing
3. **Better Maintenance** - No orphaned dependencies on missing modules
4. **Reduced Clutter** - Only essential scripts remain
5. **Logical Grouping** - Setup, processing, and generation are separate

## 🔧 No Import Changes Needed

Since most scripts are standalone with relative paths to `src/` and `data/`, no import statement updates are required. The scripts will continue to work from their new locations.