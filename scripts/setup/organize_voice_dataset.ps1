# EchoTales Enhanced - Voice Dataset Organization Script
# This script organizes voice data by gender: 12 male actors, 11 female actors
# Total: 1,012 voice samples (528 male + 484 female)

param(
    [string]$SourcePath = "audio",
    [string]$DestinationPath = "data/voice_dataset",
    [switch]$DryRun = $false,
    [switch]$CreateStructure = $false
)

Write-Host "EchoTales Enhanced - Voice Dataset Organization" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# Define paths
$ProjectRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$SourceDir = Join-Path $ProjectRoot $SourcePath
$DestDir = Join-Path $ProjectRoot $DestinationPath

Write-Host "Project Root: $ProjectRoot" -ForegroundColor Yellow
Write-Host "Source Directory: $SourceDir" -ForegroundColor Yellow
Write-Host "Destination Directory: $DestDir" -ForegroundColor Yellow

# Voice dataset configuration
$VoiceConfig = @{
    MaleActors = @{
        Count = 12
        Actors = @("Actor_01", "Actor_03", "Actor_05", "Actor_07", "Actor_09", "Actor_11", 
                  "Actor_13", "Actor_15", "Actor_17", "Actor_19", "Actor_21", "Actor_23")
        DestFolder = "male_actors"
    }
    FemaleActors = @{
        Count = 11  
        Actors = @("Actor_02", "Actor_04", "Actor_06", "Actor_08", "Actor_10", "Actor_12",
                  "Actor_14", "Actor_16", "Actor_18", "Actor_20", "Actor_22")
        DestFolder = "female_actors"
    }
    SamplesPerActor = 44
}

# Create directory structure
function Create-VoiceDatasetStructure {
    param([string]$BasePath)
    
    Write-Host "`nCreating voice dataset directory structure..." -ForegroundColor Green
    
    $Directories = @(
        "$BasePath",
        "$BasePath/male_actors",
        "$BasePath/female_actors", 
        "$BasePath/processed_samples",
        "$BasePath/processed_samples/embeddings",
        "$BasePath/processed_samples/male_processed",
        "$BasePath/processed_samples/female_processed",
        "$BasePath/metadata"
    )
    
    foreach ($Dir in $Directories) {
        $FullPath = $Dir -replace '/', '\'
        if (-not (Test-Path $FullPath)) {
            New-Item -ItemType Directory -Path $FullPath -Force | Out-Null
            Write-Host "  ✓ Created: $FullPath" -ForegroundColor Gray
        } else {
            Write-Host "  ✓ Exists: $FullPath" -ForegroundColor DarkGray
        }
    }
}

# Analyze existing actor data
function Analyze-ActorData {
    param([string]$SourcePath)
    
    Write-Host "`nAnalyzing existing actor data..." -ForegroundColor Green
    
    if (-not (Test-Path $SourcePath)) {
        Write-Host "  ⚠️  Source path does not exist: $SourcePath" -ForegroundColor Yellow
        return @()
    }
    
    $ActorFolders = Get-ChildItem -Path $SourcePath -Directory | Where-Object { $_.Name -match "Actor_\d+" }
    
    if ($ActorFolders.Count -eq 0) {
        Write-Host "  ⚠️  No Actor_* folders found in source path" -ForegroundColor Yellow
        return @()
    }
    
    $Analysis = @()
    foreach ($Actor in $ActorFolders) {
        $AudioFiles = Get-ChildItem -Path $Actor.FullName -Filter "*.wav" -File
        
        $ActorInfo = @{
            Name = $Actor.Name
            Path = $Actor.FullName
            AudioFileCount = $AudioFiles.Count
            TotalSize = ($AudioFiles | Measure-Object -Property Length -Sum).Sum / 1MB
            FirstFile = $AudioFiles | Select-Object -First 1 -ExpandProperty Name
        }
        
        $Analysis += $ActorInfo
        
        Write-Host "  📁 $($Actor.Name): $($AudioFiles.Count) files, $($ActorInfo.TotalSize.ToString('F1')) MB" -ForegroundColor Gray
        if ($AudioFiles.Count -ne 44) {
            Write-Host "    ⚠️  Expected 44 files, found $($AudioFiles.Count)" -ForegroundColor Yellow
        }
    }
    
    return $Analysis
}

# Organize actors by gender based on naming convention
function Organize-ActorsByGender {
    param([array]$ActorAnalysis)
    
    Write-Host "`nOrganizing actors by gender..." -ForegroundColor Green
    
    $MaleActors = @()
    $FemaleActors = @()
    
    foreach ($Actor in $ActorAnalysis) {
        # Check if it's an odd or even numbered actor (assuming odd=male, even=female)
        $ActorNumber = [int]($Actor.Name -replace "Actor_", "")
        
        if ($ActorNumber % 2 -eq 1) {
            # Odd numbers = Male
            $MaleActors += $Actor
            Write-Host "  👨 $($Actor.Name) -> Male (Actor $ActorNumber)" -ForegroundColor Blue
        } else {
            # Even numbers = Female  
            $FemaleActors += $Actor
            Write-Host "  👩 $($Actor.Name) -> Female (Actor $ActorNumber)" -ForegroundColor Magenta
        }
    }
    
    Write-Host "`nSummary:" -ForegroundColor Green
    Write-Host "  Male actors: $($MaleActors.Count) (Expected: 12)" -ForegroundColor Blue
    Write-Host "  Female actors: $($FemaleActors.Count) (Expected: 11)" -ForegroundColor Magenta
    Write-Host "  Total samples: $(($ActorAnalysis | Measure-Object -Property AudioFileCount -Sum).Sum)" -ForegroundColor Cyan
    
    return @{
        Male = $MaleActors
        Female = $FemaleActors
    }
}

# Copy and organize files
function Copy-VoiceFiles {
    param(
        [array]$Actors,
        [string]$DestinationFolder,
        [string]$Gender,
        [bool]$DryRun
    )
    
    Write-Host "`nCopying $Gender actor files..." -ForegroundColor Green
    
    $DestPath = Join-Path $DestDir $DestinationFolder
    
    foreach ($Actor in $Actors) {
        $ActorDestPath = Join-Path $DestPath $Actor.Name
        
        if ($DryRun) {
            Write-Host "  [DRY RUN] Would copy $($Actor.Name) to $ActorDestPath" -ForegroundColor Yellow
            continue
        }
        
        # Create actor-specific directory
        if (-not (Test-Path $ActorDestPath)) {
            New-Item -ItemType Directory -Path $ActorDestPath -Force | Out-Null
        }
        
        # Copy audio files
        $SourceFiles = Get-ChildItem -Path $Actor.Path -Filter "*.wav" -File
        
        Write-Host "  📁 $($Actor.Name): Copying $($SourceFiles.Count) files..." -ForegroundColor Gray
        
        foreach ($File in $SourceFiles) {
            $DestFile = Join-Path $ActorDestPath $File.Name
            Copy-Item -Path $File.FullName -Destination $DestFile -Force
        }
        
        Write-Host "    ✓ Copied $($SourceFiles.Count) files to $ActorDestPath" -ForegroundColor DarkGreen
    }
}

# Create metadata file
function Create-VoiceDatasetMetadata {
    param([hashtable]$OrganizedActors, [string]$MetadataPath)
    
    Write-Host "`nCreating metadata file..." -ForegroundColor Green
    
    $Metadata = @{
        CreatedDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Version = "1.0"
        Description = "EchoTales Enhanced Voice Dataset"
        TotalActors = ($OrganizedActors.Male.Count + $OrganizedActors.Female.Count)
        MaleActors = @{
            Count = $OrganizedActors.Male.Count
            ExpectedCount = 12
            SamplesPerActor = 44
            TotalSamples = ($OrganizedActors.Male | Measure-Object -Property AudioFileCount -Sum).Sum
            Actors = ($OrganizedActors.Male | ForEach-Object { 
                @{
                    Name = $_.Name
                    SampleCount = $_.AudioFileCount
                    SizeMB = [math]::Round($_.TotalSize, 2)
                }
            })
        }
        FemaleActors = @{
            Count = $OrganizedActors.Female.Count
            ExpectedCount = 11
            SamplesPerActor = 44
            TotalSamples = ($OrganizedActors.Female | Measure-Object -Property AudioFileCount -Sum).Sum
            Actors = ($OrganizedActors.Female | ForEach-Object {
                @{
                    Name = $_.Name
                    SampleCount = $_.AudioFileCount
                    SizeMP = [math]::Round($_.TotalSize, 2)
                }
            })
        }
        AudioFormat = @{
            SampleRate = "22050 Hz"
            Channels = "Mono"
            BitDepth = "16-bit"
            Format = "WAV"
        }
    }
    
    $MetadataFile = Join-Path $MetadataPath "voice_dataset_metadata.json"
    $Metadata | ConvertTo-Json -Depth 5 | Out-File -FilePath $MetadataFile -Encoding UTF8
    
    Write-Host "  ✓ Metadata saved to: $MetadataFile" -ForegroundColor DarkGreen
    
    # Also create a summary report
    $SummaryFile = Join-Path $MetadataPath "dataset_summary.txt"
    $Summary = @"
EchoTales Enhanced Voice Dataset Summary
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Dataset Overview:
- Total Actors: $($Metadata.TotalActors)
- Male Actors: $($Metadata.MaleActors.Count) (Expected: 12)
- Female Actors: $($Metadata.FemaleActors.Count) (Expected: 11)
- Total Audio Samples: $($Metadata.MaleActors.TotalSamples + $Metadata.FemaleActors.TotalSamples)
- Expected Total: 1,012 samples (12×44 + 11×44)

Male Actors ($($Metadata.MaleActors.Count)):
$($Metadata.MaleActors.Actors | ForEach-Object { "  - $($_.Name): $($_.SampleCount) samples, $($_.SizeMP) MB" } | Out-String)

Female Actors ($($Metadata.FemaleActors.Count)):
$($Metadata.FemaleActors.Actors | ForEach-Object { "  - $($_.Name): $($_.SampleCount) samples, $($_.SizeMP) MB" } | Out-String)

Audio Format:
- Sample Rate: 22050 Hz
- Channels: Mono
- Bit Depth: 16-bit
- Format: WAV

Directory Structure:
- Male actors: data/voice_dataset/male_actors/
- Female actors: data/voice_dataset/female_actors/
- Processed samples: data/voice_dataset/processed_samples/
- Metadata: data/voice_dataset/metadata/
"@
    
    $Summary | Out-File -FilePath $SummaryFile -Encoding UTF8
    Write-Host "  ✓ Summary report saved to: $SummaryFile" -ForegroundColor DarkGreen
}

# Main execution
try {
    Write-Host "`nStarting voice dataset organization..." -ForegroundColor Cyan
    
    # Create directory structure if requested
    if ($CreateStructure) {
        Create-VoiceDatasetStructure -BasePath $DestDir
    }
    
    # Analyze existing data
    $ActorAnalysis = Analyze-ActorData -SourcePath $SourceDir
    
    if ($ActorAnalysis.Count -eq 0) {
        Write-Host "`n❌ No actor data found in source directory!" -ForegroundColor Red
        Write-Host "Please ensure your actor folders (Actor_01, Actor_02, etc.) are in: $SourceDir" -ForegroundColor Yellow
        exit 1
    }
    
    # Organize by gender
    $OrganizedActors = Organize-ActorsByGender -ActorAnalysis $ActorAnalysis
    
    # Validate counts
    $ExpectedMale = 12
    $ExpectedFemale = 11
    
    if ($OrganizedActors.Male.Count -ne $ExpectedMale) {
        Write-Host "⚠️  Warning: Expected $ExpectedMale male actors, found $($OrganizedActors.Male.Count)" -ForegroundColor Yellow
    }
    
    if ($OrganizedActors.Female.Count -ne $ExpectedFemale) {
        Write-Host "⚠️  Warning: Expected $ExpectedFemale female actors, found $($OrganizedActors.Female.Count)" -ForegroundColor Yellow
    }
    
    if (-not $DryRun) {
        # Ensure destination structure exists
        Create-VoiceDatasetStructure -BasePath $DestDir
        
        # Copy files
        Copy-VoiceFiles -Actors $OrganizedActors.Male -DestinationFolder "male_actors" -Gender "Male" -DryRun $false
        Copy-VoiceFiles -Actors $OrganizedActors.Female -DestinationFolder "female_actors" -Gender "Female" -DryRun $false
        
        # Create metadata
        $MetadataPath = Join-Path $DestDir "metadata"
        Create-VoiceDatasetMetadata -OrganizedActors $OrganizedActors -MetadataPath $MetadataPath
        
        Write-Host "`n✅ Voice dataset organization complete!" -ForegroundColor Green
        Write-Host "✅ Male actors organized in: $DestDir\male_actors" -ForegroundColor Green
        Write-Host "✅ Female actors organized in: $DestDir\female_actors" -ForegroundColor Green
    } else {
        Write-Host "`n[DRY RUN] Organization would be complete. Use -DryRun:`$false to execute." -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "`n❌ Error during voice dataset organization:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Display final summary
Write-Host "`n" + "="*50 -ForegroundColor Cyan
Write-Host "VOICE DATASET ORGANIZATION SUMMARY" -ForegroundColor Cyan
Write-Host "="*50 -ForegroundColor Cyan
Write-Host "Total Actors Expected: 23 (12 Male + 11 Female)" -ForegroundColor White
Write-Host "Total Samples Expected: 1,012 (23 × 44)" -ForegroundColor White
Write-Host "Male Actors Found: $($OrganizedActors.Male.Count)" -ForegroundColor Blue
Write-Host "Female Actors Found: $($OrganizedActors.Female.Count)" -ForegroundColor Magenta
Write-Host "Total Samples Found: $(($ActorAnalysis | Measure-Object -Property AudioFileCount -Sum).Sum)" -ForegroundColor Cyan
Write-Host "="*50 -ForegroundColor Cyan