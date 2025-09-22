# EchoTales Enhanced - Numbered Voice Dataset Organization
# Organizes voice data: 12 male actors (1-12) and 11 female actors (1-11)

param(
    [string]$SourcePath = "audio",
    [string]$DestinationPath = "data/voice_dataset",
    [switch]$DryRun = $false
)

Write-Host "EchoTales Enhanced - Numbered Voice Dataset Organization" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# Define paths
$ProjectRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$SourceDir = Join-Path $ProjectRoot $SourcePath
$DestDir = Join-Path $ProjectRoot $DestinationPath

Write-Host "Source Directory: $SourceDir" -ForegroundColor Yellow
Write-Host "Destination Directory: $DestDir" -ForegroundColor Yellow

function Analyze-NumberedVoiceData {
    Write-Host "`nAnalyzing voice data..." -ForegroundColor Green
    
    $MalePath = Join-Path $SourceDir "male"
    $FemalePath = Join-Path $SourceDir "female"
    
    $Analysis = @{
        Male = @()
        Female = @()
    }
    
    # Analyze male actors
    if (Test-Path $MalePath) {
        $MaleFolders = Get-ChildItem -Path $MalePath -Directory | Where-Object { $_.Name -match "^\d+$" }
        Write-Host "Found $($MaleFolders.Count) male actor folders" -ForegroundColor Blue
        
        foreach ($Actor in $MaleFolders) {
            $AudioFiles = Get-ChildItem -Path $Actor.FullName -Filter "*.wav" -File
            $ActorInfo = @{
                Number = [int]$Actor.Name
                Name = "Male_$($Actor.Name)"
                Path = $Actor.FullName
                AudioFileCount = $AudioFiles.Count
                TotalSize = ($AudioFiles | Measure-Object -Property Length -Sum).Sum / 1MB
            }
            $Analysis.Male += $ActorInfo
            Write-Host "  Male Actor $($Actor.Name): $($AudioFiles.Count) files, $($ActorInfo.TotalSize.ToString('F1')) MB" -ForegroundColor Blue
        }
    } else {
        Write-Host "Male actors path not found: $MalePath" -ForegroundColor Yellow
    }
    
    # Analyze female actors
    if (Test-Path $FemalePath) {
        $FemaleFolders = Get-ChildItem -Path $FemalePath -Directory | Where-Object { $_.Name -match "^\d+$" }
        Write-Host "Found $($FemaleFolders.Count) female actor folders" -ForegroundColor Magenta
        
        foreach ($Actor in $FemaleFolders) {
            $AudioFiles = Get-ChildItem -Path $Actor.FullName -Filter "*.wav" -File
            $ActorInfo = @{
                Number = [int]$Actor.Name
                Name = "Female_$($Actor.Name)"
                Path = $Actor.FullName
                AudioFileCount = $AudioFiles.Count
                TotalSize = ($AudioFiles | Measure-Object -Property Length -Sum).Sum / 1MB
            }
            $Analysis.Female += $ActorInfo
            Write-Host "  Female Actor $($Actor.Name): $($AudioFiles.Count) files, $($ActorInfo.TotalSize.ToString('F1')) MB" -ForegroundColor Magenta
        }
    } else {
        Write-Host "Female actors path not found: $FemalePath" -ForegroundColor Yellow
    }
    
    return $Analysis
}

function Copy-NumberedVoiceFiles {
    param(
        [array]$Actors,
        [string]$DestinationFolder,
        [string]$Gender,
        [bool]$DryRun
    )
    
    Write-Host "`nProcessing $Gender actors..." -ForegroundColor Green
    
    $DestPath = Join-Path $DestDir $DestinationFolder
    
    if (-not (Test-Path $DestPath)) {
        New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
    }
    
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
        
        Write-Host "  Copying $($Actor.Name): $($SourceFiles.Count) files..." -ForegroundColor Gray
        
        foreach ($File in $SourceFiles) {
            $DestFile = Join-Path $ActorDestPath $File.Name
            Copy-Item -Path $File.FullName -Destination $DestFile -Force
        }
        
        Write-Host "    Completed: $($SourceFiles.Count) files to $ActorDestPath" -ForegroundColor DarkGreen
    }
}

function Create-NumberedVoiceMetadata {
    param([hashtable]$Analysis, [string]$MetadataPath)
    
    Write-Host "`nCreating metadata..." -ForegroundColor Green
    
    if (-not (Test-Path $MetadataPath)) {
        New-Item -ItemType Directory -Path $MetadataPath -Force | Out-Null
    }
    
    $Metadata = @{
        CreatedDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Version = "1.0"
        Description = "EchoTales Enhanced Numbered Voice Dataset"
        NamingScheme = "Numbered (1-12 male, 1-11 female)"
        TotalActors = ($Analysis.Male.Count + $Analysis.Female.Count)
        MaleActors = @{
            Count = $Analysis.Male.Count
            ExpectedCount = 12
            Range = "1-12"
            SamplesPerActor = 44
            TotalSamples = ($Analysis.Male | Measure-Object -Property AudioFileCount -Sum).Sum
            Actors = ($Analysis.Male | Sort-Object Number | ForEach-Object { 
                @{
                    Number = $_.Number
                    Name = $_.Name
                    SampleCount = $_.AudioFileCount
                    SizeMB = [math]::Round($_.TotalSize, 2)
                }
            })
        }
        FemaleActors = @{
            Count = $Analysis.Female.Count
            ExpectedCount = 11
            Range = "1-11"
            SamplesPerActor = 44
            TotalSamples = ($Analysis.Female | Measure-Object -Property AudioFileCount -Sum).Sum
            Actors = ($Analysis.Female | Sort-Object Number | ForEach-Object {
                @{
                    Number = $_.Number
                    Name = $_.Name
                    SampleCount = $_.AudioFileCount
                    SizeMB = [math]::Round($_.TotalSize, 2)
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
    
    $MetadataFile = Join-Path $MetadataPath "numbered_voice_dataset_metadata.json"
    $Metadata | ConvertTo-Json -Depth 5 | Out-File -FilePath $MetadataFile -Encoding UTF8
    
    Write-Host "  Metadata saved to: $MetadataFile" -ForegroundColor DarkGreen
    
    # Create summary report
    $SummaryFile = Join-Path $MetadataPath "numbered_dataset_summary.txt"
    $Summary = @"
EchoTales Enhanced - Numbered Voice Dataset Summary
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Dataset Overview:
- Naming Scheme: Numbered folders (1-12 male, 1-11 female)  
- Total Actors: $($Metadata.TotalActors)
- Male Actors: $($Metadata.MaleActors.Count) (Expected: 12, Range: 1-12)
- Female Actors: $($Metadata.FemaleActors.Count) (Expected: 11, Range: 1-11)
- Total Audio Samples: $($Metadata.MaleActors.TotalSamples + $Metadata.FemaleActors.TotalSamples)
- Expected Total: 1,012 samples (12×44 + 11×44)

Male Actors ($($Metadata.MaleActors.Count)):
$($Metadata.MaleActors.Actors | ForEach-Object { "  - Actor $($_.Number): $($_.SampleCount) samples, $($_.SizeMB) MB" } | Out-String)

Female Actors ($($Metadata.FemaleActors.Count)):
$($Metadata.FemaleActors.Actors | ForEach-Object { "  - Actor $($_.Number): $($_.SampleCount) samples, $($_.SizeMB) MB" } | Out-String)

Source Structure:
- Male actors: audio/male/1/ through audio/male/12/
- Female actors: audio/female/1/ through audio/female/11/

Destination Structure:  
- Male actors: data/voice_dataset/male_actors/
- Female actors: data/voice_dataset/female_actors/
- Processed samples: data/voice_dataset/processed_samples/
- Metadata: data/voice_dataset/metadata/
"@
    
    $Summary | Out-File -FilePath $SummaryFile -Encoding UTF8
    Write-Host "  Summary saved to: $SummaryFile" -ForegroundColor DarkGreen
}

# Main execution
try {
    Write-Host "`nStarting numbered voice dataset organization..." -ForegroundColor Cyan
    
    # Analyze data
    $Analysis = Analyze-NumberedVoiceData
    
    if ($Analysis.Male.Count -eq 0 -and $Analysis.Female.Count -eq 0) {
        Write-Host "`nNo voice data found!" -ForegroundColor Red
        Write-Host "Please ensure your structure is:" -ForegroundColor Yellow
        Write-Host "  audio/male/1/ through audio/male/12/" -ForegroundColor Blue  
        Write-Host "  audio/female/1/ through audio/female/11/" -ForegroundColor Magenta
        exit 1
    }
    
    # Display summary
    Write-Host "`nDataset Summary:" -ForegroundColor Green
    Write-Host "  Male actors found: $($Analysis.Male.Count) (Expected: 12)" -ForegroundColor Blue
    Write-Host "  Female actors found: $($Analysis.Female.Count) (Expected: 11)" -ForegroundColor Magenta
    Write-Host "  Total samples: $(($Analysis.Male | Measure-Object -Property AudioFileCount -Sum).Sum + ($Analysis.Female | Measure-Object -Property AudioFileCount -Sum).Sum)" -ForegroundColor Cyan
    
    if (-not $DryRun) {
        # Copy files  
        Copy-NumberedVoiceFiles -Actors $Analysis.Male -DestinationFolder "male_actors" -Gender "Male" -DryRun $false
        Copy-NumberedVoiceFiles -Actors $Analysis.Female -DestinationFolder "female_actors" -Gender "Female" -DryRun $false
        
        # Create metadata
        $MetadataPath = Join-Path $DestDir "metadata"
        Create-NumberedVoiceMetadata -Analysis $Analysis -MetadataPath $MetadataPath
        
        Write-Host "`nOrganization complete!" -ForegroundColor Green
        Write-Host "Male actors organized in: $DestDir\male_actors" -ForegroundColor Blue
        Write-Host "Female actors organized in: $DestDir\female_actors" -ForegroundColor Magenta
    } else {
        Write-Host "`n[DRY RUN] Use -DryRun:`$false to execute the organization." -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "`nError during organization:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host "`n" + "="*60 -ForegroundColor Cyan
Write-Host "NUMBERED VOICE DATASET ORGANIZATION COMPLETE" -ForegroundColor Cyan  
Write-Host "="*60 -ForegroundColor Cyan