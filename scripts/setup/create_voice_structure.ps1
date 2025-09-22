# Quick Voice Dataset Structure Creator
# Creates the directory structure for organizing voice data by gender

Write-Host "Creating Voice Dataset Directory Structure..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$ProjectRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent

# Directory structure for voice dataset
$Directories = @(
    # Main voice dataset directories
    "data\voice_dataset",
    "data\voice_dataset\male_actors", 
    "data\voice_dataset\female_actors",
    "data\voice_dataset\processed_samples",
    "data\voice_dataset\processed_samples\embeddings",
    "data\voice_dataset\processed_samples\male_processed",
    "data\voice_dataset\processed_samples\female_processed",
    "data\voice_dataset\metadata",
    
    # Additional audio directories
    "data\raw\audio_samples",
    "data\raw\audio_samples\source_recordings",
    "data\raw\audio_samples\reference_voices",
    "data\raw\audio_samples\temp_uploads",
    "data\raw\audio_samples\external_datasets",
    
    # Output directories (already exist but ensure they're there)
    "data\output\audio\characters",
    "data\output\audio\chapters", 
    "data\output\audio\full_books",
    
    # Cache directories
    "cache\audio",
    "cache\audio\processing",
    "cache\audio\voice_models",
    "cache\audio\temp_generations",
    
    # Model directories
    "models\voice_cloning",
    "models\voice_cloning\base_models",
    "models\voice_cloning\custom_models", 
    "models\voice_cloning\embeddings",
    "models\voice_cloning\checkpoints"
)

Write-Host "Creating directories..." -ForegroundColor Green

foreach ($Dir in $Directories) {
    $FullPath = Join-Path $ProjectRoot $Dir
    
    if (-not (Test-Path $FullPath)) {
        New-Item -ItemType Directory -Path $FullPath -Force | Out-Null
        Write-Host "  ✓ Created: $Dir" -ForegroundColor Gray
    } else {
        Write-Host "  ✓ Exists: $Dir" -ForegroundColor DarkGray
    }
}

# Create placeholder README files
$ReadmeFiles = @{
    "data\voice_dataset\README.md" = @"
# Voice Dataset

This directory contains organized voice data for EchoTales Enhanced.

## Structure:
- `male_actors/` - 12 male actor voice samples (44 samples each)
- `female_actors/` - 11 female actor voice samples (44 samples each) 
- `processed_samples/` - Processed and cleaned voice data
- `metadata/` - Dataset metadata and statistics

## Expected Dataset:
- Total actors: 23 (12 male + 11 female)
- Samples per actor: 44
- Total samples: 1,012
- Format: WAV, 22050 Hz, Mono, 16-bit

## Usage:
Run the organization script to populate this structure:
```powershell
.\scripts\setup\organize_voice_dataset.ps1
```
"@

    "data\raw\audio_samples\README.md" = @"
# Raw Audio Samples

Store raw, unprocessed audio files here before organizing them into the voice dataset.

## Subdirectories:
- `source_recordings/` - Original voice recordings
- `reference_voices/` - Reference audio for specific characters
- `temp_uploads/` - Temporary uploaded audio files
- `external_datasets/` - Third-party audio datasets
"@

    "cache\audio\README.md" = @"
# Audio Cache

Temporary storage for audio processing and generation.

## Subdirectories:
- `processing/` - Files being processed
- `voice_models/` - Cached voice models
- `temp_generations/` - Temporary generated audio

**Note:** Files in this directory are temporary and can be safely deleted.
"@
}

Write-Host "`nCreating documentation..." -ForegroundColor Green

foreach ($File in $ReadmeFiles.Keys) {
    $FullPath = Join-Path $ProjectRoot $File
    $Content = $ReadmeFiles[$File]
    
    if (-not (Test-Path $FullPath)) {
        $Content | Out-File -FilePath $FullPath -Encoding UTF8
        Write-Host "  ✓ Created: $File" -ForegroundColor Gray
    }
}

Write-Host "`nVoice dataset directory structure created!" -ForegroundColor Green
Write-Host "`nDirectory Structure:" -ForegroundColor Yellow
Write-Host "data/voice_dataset/male_actors/     (12 actors, 44 samples each)" -ForegroundColor Blue
Write-Host "data/voice_dataset/female_actors/   (11 actors, 44 samples each)" -ForegroundColor Magenta
Write-Host "data/voice_dataset/processed_samples/" -ForegroundColor White
Write-Host "data/voice_dataset/metadata/" -ForegroundColor White
Write-Host "data/raw/audio_samples/" -ForegroundColor White
Write-Host "cache/audio/" -ForegroundColor White
Write-Host "models/voice_cloning/" -ForegroundColor White

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "1. Place your Actor_01-Actor_24 folders in the 'audio/' directory" -ForegroundColor White
Write-Host "2. Run: .\scripts\setup\organize_voice_dataset.ps1" -ForegroundColor White
Write-Host "3. This will organize your 1,012 voice samples by gender" -ForegroundColor White

Write-Host "`nExpected Dataset:" -ForegroundColor Yellow
Write-Host "Male Actors (odd numbers): Actor_01, 03, 05, 07, 09, 11, 13, 15, 17, 19, 21, 23" -ForegroundColor Blue
Write-Host "Female Actors (even numbers): Actor_02, 04, 06, 08, 10, 12, 14, 16, 18, 20, 22" -ForegroundColor Magenta
Write-Host "Total: 1,012 samples (528 male + 484 female)" -ForegroundColor Cyan
