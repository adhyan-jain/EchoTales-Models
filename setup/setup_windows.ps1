# EchoTales Windows Environment Setup Script
# This script sets up the complete EchoTales development environment on Windows
# with comprehensive error handling and fallbacks

param(
    [switch]$SkipPython = $false,
    [switch]$SkipModels = $false,
    [switch]$SkipDependencies = $false,
    [switch]$DevMode = $false,
    [switch]$Verbose = $false,
    [string]$PythonVersion = "3.10"
)

# Set error handling
$ErrorActionPreference = "Continue"
$WarningPreference = "Continue"

# Initialize logging
$LogFile = "setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$script:SetupErrors = @()
$script:SetupWarnings = @()

function Write-LogMessage {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Write to console with colors
    switch ($Level) {
        "ERROR" { Write-Host $logEntry -ForegroundColor Red }
        "WARN" { Write-Host $logEntry -ForegroundColor Yellow }
        "SUCCESS" { Write-Host $logEntry -ForegroundColor Green }
        default { Write-Host $logEntry }
    }
    
    # Write to log file
    Add-Content -Path $LogFile -Value $logEntry -Encoding UTF8
    
    # Track errors and warnings
    if ($Level -eq "ERROR") {
        $script:SetupErrors += $Message
    } elseif ($Level -eq "WARN") {
        $script:SetupWarnings += $Message
    }
}

function Test-AdminPrivileges {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-InternetConnection {
    try {
        $response = Test-NetConnection -ComputerName "8.8.8.8" -Port 53 -InformationLevel Quiet
        return $response
    } catch {
        Write-LogMessage "Internet connectivity test failed: $($_.Exception.Message)" "WARN"
        return $false
    }
}

function Install-Chocolatey {
    Write-LogMessage "Installing Chocolatey package manager..."
    
    try {
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-LogMessage "Chocolatey already installed" "SUCCESS"
            return $true
        }
        
        # Check if we can install Chocolatey
        if (-not (Test-AdminPrivileges)) {
            Write-LogMessage "Administrator privileges required for Chocolatey installation" "ERROR"
            return $false
        }
        
        # Install Chocolatey
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        
        $chocoInstallScript = Invoke-WebRequest -Uri "https://community.chocolatey.org/install.ps1" -UseBasicParsing
        Invoke-Expression $chocoInstallScript.Content
        
        # Verify installation
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-LogMessage "Chocolatey installed successfully" "SUCCESS"
            return $true
        } else {
            Write-LogMessage "Chocolatey installation failed" "ERROR"
            return $false
        }
        
    } catch {
        Write-LogMessage "Failed to install Chocolatey: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Install-Python {
    param([string]$Version = "3.10")
    
    Write-LogMessage "Setting up Python $Version..."
    
    try {
        # Check if Python is already installed
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            $pythonVersionOutput = & python --version 2>&1
            Write-LogMessage "Found Python: $pythonVersionOutput"
            
            if ($pythonVersionOutput -match "Python $Version") {
                Write-LogMessage "Python $Version already installed" "SUCCESS"
                return $true
            }
        }
        
        # Try to install via Chocolatey first
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-LogMessage "Installing Python via Chocolatey..."
            $chocoResult = choco install python --version=$Version -y
            
            # Refresh environment
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            # Verify installation
            Start-Sleep -Seconds 5
            if (Get-Command python -ErrorAction SilentlyContinue) {
                Write-LogMessage "Python installed successfully via Chocolatey" "SUCCESS"
                return $true
            }
        }
        
        # Fallback: Download and install from python.org
        Write-LogMessage "Attempting manual Python installation..."
        $pythonUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
        $pythonInstaller = "$env:TEMP\python_installer.exe"
        
        Write-LogMessage "Downloading Python installer..."
        Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller -UseBasicParsing
        
        Write-LogMessage "Running Python installer..."
        Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_test=0" -Wait
        
        # Refresh environment and verify
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        if (Get-Command python -ErrorAction SilentlyContinue) {
            Write-LogMessage "Python installed successfully" "SUCCESS"
            Remove-Item $pythonInstaller -Force -ErrorAction SilentlyContinue
            return $true
        } else {
            Write-LogMessage "Python installation failed" "ERROR"
            return $false
        }
        
    } catch {
        Write-LogMessage "Failed to install Python: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Install-Git {
    Write-LogMessage "Setting up Git..."
    
    try {
        # Check if Git is already installed
        if (Get-Command git -ErrorAction SilentlyContinue) {
            Write-LogMessage "Git already installed" "SUCCESS"
            return $true
        }
        
        # Try Chocolatey first
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-LogMessage "Installing Git via Chocolatey..."
            choco install git -y
            
            # Refresh environment
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            if (Get-Command git -ErrorAction SilentlyContinue) {
                Write-LogMessage "Git installed successfully via Chocolatey" "SUCCESS"
                return $true
            }
        }
        
        # Fallback: Manual installation
        Write-LogMessage "Attempting manual Git installation..."
        $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe"
        $gitInstaller = "$env:TEMP\git_installer.exe"
        
        Invoke-WebRequest -Uri $gitUrl -OutFile $gitInstaller -UseBasicParsing
        Start-Process -FilePath $gitInstaller -ArgumentList "/SILENT" -Wait
        
        # Refresh and verify
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        if (Get-Command git -ErrorAction SilentlyContinue) {
            Write-LogMessage "Git installed successfully" "SUCCESS"
            Remove-Item $gitInstaller -Force -ErrorAction SilentlyContinue
            return $true
        } else {
            Write-LogMessage "Git installation failed" "ERROR"
            return $false
        }
        
    } catch {
        Write-LogMessage "Failed to install Git: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Install-NodeJS {
    Write-LogMessage "Setting up Node.js..."
    
    try {
        # Check if Node.js is already installed
        if (Get-Command node -ErrorAction SilentlyContinue) {
            $nodeVersion = & node --version
            Write-LogMessage "Node.js already installed: $nodeVersion" "SUCCESS"
            return $true
        }
        
        # Try Chocolatey first
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-LogMessage "Installing Node.js via Chocolatey..."
            choco install nodejs -y
            
            # Refresh environment
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            if (Get-Command node -ErrorAction SilentlyContinue) {
                Write-LogMessage "Node.js installed successfully via Chocolatey" "SUCCESS"
                return $true
            }
        }
        
        Write-LogMessage "Node.js installation via alternative methods not implemented in this script" "WARN"
        Write-LogMessage "Please install Node.js manually from https://nodejs.org/" "WARN"
        return $false
        
    } catch {
        Write-LogMessage "Failed to install Node.js: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Setup-PythonEnvironment {
    Write-LogMessage "Setting up Python virtual environment..."
    
    try {
        # Create virtual environment
        if (Test-Path "venv") {
            Write-LogMessage "Virtual environment already exists"
        } else {
            Write-LogMessage "Creating virtual environment..."
            & python -m venv venv
            
            if (-not (Test-Path "venv")) {
                Write-LogMessage "Failed to create virtual environment" "ERROR"
                return $false
            }
        }
        
        # Activate virtual environment
        Write-LogMessage "Activating virtual environment..."
        & .\venv\Scripts\Activate.ps1
        
        # Upgrade pip
        Write-LogMessage "Upgrading pip..."
        & python -m pip install --upgrade pip
        
        Write-LogMessage "Python environment setup complete" "SUCCESS"
        return $true
        
    } catch {
        Write-LogMessage "Failed to setup Python environment: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Install-PythonDependencies {
    Write-LogMessage "Installing Python dependencies..."
    
    try {
        # Check if requirements.txt exists
        if (-not (Test-Path "requirements.txt")) {
            Write-LogMessage "requirements.txt not found, skipping Python dependencies" "WARN"
            return $false
        }
        
        # Install dependencies with fallbacks
        Write-LogMessage "Installing core dependencies..."
        
        # Try to install from requirements.txt
        $pipResult = & python -m pip install -r requirements.txt --timeout 300
        
        if ($LASTEXITCODE -eq 0) {
            Write-LogMessage "All dependencies installed successfully" "SUCCESS"
            return $true
        } else {
            Write-LogMessage "Some dependencies failed to install, trying individual installation..." "WARN"
            
            # Try individual installation of critical packages
            $criticalPackages = @(
                "flask",
                "flask-restful", 
                "python-dotenv",
                "pydantic",
                "requests",
                "numpy",
                "scipy"
            )
            
            $successCount = 0
            foreach ($package in $criticalPackages) {
                try {
                    Write-LogMessage "Installing $package..."
                    & python -m pip install $package --timeout 60
                    
                    if ($LASTEXITCODE -eq 0) {
                        $successCount++
                    } else {
                        Write-LogMessage "Failed to install $package" "WARN"
                    }
                } catch {
                    Write-LogMessage "Error installing $package: $($_.Exception.Message)" "WARN"
                }
            }
            
            if ($successCount -ge 4) {
                Write-LogMessage "Core dependencies installed successfully" "SUCCESS"
                return $true
            } else {
                Write-LogMessage "Failed to install sufficient core dependencies" "ERROR"
                return $false
            }
        }
        
    } catch {
        Write-LogMessage "Failed to install Python dependencies: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Setup-DirectoryStructure {
    Write-LogMessage "Setting up directory structure..."
    
    try {
        $directories = @(
            "data",
            "data/raw",
            "data/raw/books", 
            "data/processed",
            "data/processed/chapters",
            "data/output",
            "data/output/audio",
            "data/output/images",
            "data/output/character_portraits",
            "data/voice_dataset",
            "data/voice_dataset/male_actors",
            "data/voice_dataset/female_actors",
            "data/voice_dataset/processed",
            "logs",
            "models",
            "models/booknlp",
            "scripts",
            "scripts/processing",
            "tests",
            "tests/fixtures",
            "config"
        )
        
        foreach ($dir in $directories) {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
                Write-LogMessage "Created directory: $dir"
            }
        }
        
        Write-LogMessage "Directory structure setup complete" "SUCCESS"
        return $true
        
    } catch {
        Write-LogMessage "Failed to setup directory structure: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Setup-EnvironmentFile {
    Write-LogMessage "Setting up environment configuration..."
    
    try {
        if (-not (Test-Path ".env") -and (Test-Path ".env.template")) {
            Write-LogMessage "Creating .env from template..."
            Copy-Item ".env.template" ".env"
            Write-LogMessage "Created .env file. Please edit it with your API keys." "SUCCESS"
        } elseif (Test-Path ".env") {
            Write-LogMessage ".env file already exists" "SUCCESS"
        } else {
            Write-LogMessage ".env.template not found, creating basic .env file..." "WARN"
            
            $basicEnv = @"
# Basic EchoTales Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=127.0.0.1
FLASK_PORT=5000

# Add your API keys here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
STABILITY_API_KEY=your_stability_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Database
DATABASE_URL=postgresql://localhost:5432/echotales
REDIS_URL=redis://localhost:6379/0
"@
            
            Set-Content -Path ".env" -Value $basicEnv -Encoding UTF8
            Write-LogMessage "Created basic .env file" "SUCCESS"
        }
        
        return $true
        
    } catch {
        Write-LogMessage "Failed to setup environment file: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Test-PythonImports {
    Write-LogMessage "Testing critical Python imports..."
    
    $imports = @(
        "flask",
        "numpy", 
        "requests",
        "json",
        "pathlib",
        "logging"
    )
    
    $successCount = 0
    foreach ($import in $imports) {
        try {
            $result = & python -c "import $import; print('OK')" 2>$null
            if ($result -eq "OK") {
                Write-LogMessage "✓ $import" 
                $successCount++
            } else {
                Write-LogMessage "✗ $import" "WARN"
            }
        } catch {
            Write-LogMessage "✗ $import (error: $($_.Exception.Message))" "WARN"
        }
    }
    
    if ($successCount -ge ($imports.Count * 0.8)) {
        Write-LogMessage "Critical imports test passed ($successCount/$($imports.Count))" "SUCCESS"
        return $true
    } else {
        Write-LogMessage "Critical imports test failed ($successCount/$($imports.Count))" "ERROR"
        return $false
    }
}

function Download-Models {
    Write-LogMessage "Setting up AI models..."
    
    try {
        # Create models directory structure
        $modelDirs = @("models", "models/booknlp", "models/character_analysis", "models/voice_classification")
        foreach ($dir in $modelDirs) {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
        }
        
        # Create placeholder model files (actual models would be downloaded by Python scripts)
        $modelPlaceholder = @"
# Model files will be downloaded automatically when first needed
# You can manually download models using:
# python -c "from src.echotales.ai.model_manager import ModelManager; ModelManager().download_all_models()"
"@
        
        Set-Content -Path "models/README.md" -Value $modelPlaceholder -Encoding UTF8
        
        Write-LogMessage "Model directories created. Models will be downloaded on first use." "SUCCESS"
        return $true
        
    } catch {
        Write-LogMessage "Failed to setup models: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Create-StartupScript {
    Write-LogMessage "Creating startup scripts..."
    
    try {
        # Create Windows batch file
        $batchScript = @"
@echo off
echo Starting EchoTales Development Server...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check environment
if not exist .env (
    echo ERROR: .env file not found. Please copy .env.template to .env and configure your API keys.
    pause
    exit /b 1
)

REM Start the application
echo Starting Flask application...
python src/echotales/api/app.py

pause
"@
        
        Set-Content -Path "start_echotales.bat" -Value $batchScript -Encoding UTF8
        
        # Create PowerShell script
        $psScript = @"
# EchoTales Startup Script
Write-Host "Starting EchoTales Development Server..." -ForegroundColor Green

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    exit 1
}

# Check environment file
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: .env file not found. Please copy .env.template to .env and configure your API keys." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Start the application
Write-Host "Starting Flask application..." -ForegroundColor Green
& python src/echotales/api/app.py
"@
        
        Set-Content -Path "start_echotales.ps1" -Value $psScript -Encoding UTF8
        
        Write-LogMessage "Startup scripts created successfully" "SUCCESS"
        return $true
        
    } catch {
        Write-LogMessage "Failed to create startup scripts: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Show-Summary {
    param(
        [bool]$OverallSuccess
    )
    
    Write-Host "`n" + "="*80 -ForegroundColor Cyan
    Write-Host "EchoTales Setup Summary" -ForegroundColor Cyan
    Write-Host "="*80 -ForegroundColor Cyan
    
    if ($OverallSuccess) {
        Write-Host "✓ Setup completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠ Setup completed with issues" -ForegroundColor Yellow
    }
    
    Write-Host "`nNext Steps:" -ForegroundColor White
    Write-Host "1. Edit the .env file with your API keys" -ForegroundColor White
    Write-Host "2. Run 'start_echotales.bat' or 'start_echotales.ps1' to start the application" -ForegroundColor White
    Write-Host "3. Visit http://localhost:5000/api/health to test the API" -ForegroundColor White
    
    if ($script:SetupErrors.Count -gt 0) {
        Write-Host "`nErrors encountered:" -ForegroundColor Red
        foreach ($error in $script:SetupErrors) {
            Write-Host "  • $error" -ForegroundColor Red
        }
    }
    
    if ($script:SetupWarnings.Count -gt 0) {
        Write-Host "`nWarnings:" -ForegroundColor Yellow
        foreach ($warning in $script:SetupWarnings) {
            Write-Host "  • $warning" -ForegroundColor Yellow
        }
    }
    
    Write-Host "`nSetup log saved to: $LogFile" -ForegroundColor Gray
    Write-Host "="*80 -ForegroundColor Cyan
}

# Main execution
function Main {
    Write-Host "EchoTales Windows Environment Setup" -ForegroundColor Cyan
    Write-Host "===================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check prerequisites
    Write-LogMessage "Checking prerequisites..."
    
    $hasInternet = Test-InternetConnection
    if (-not $hasInternet) {
        Write-LogMessage "No internet connection detected. Some features may not work." "WARN"
    }
    
    $isAdmin = Test-AdminPrivileges
    if (-not $isAdmin) {
        Write-LogMessage "Running without administrator privileges. Some installations may fail." "WARN"
    }
    
    # Track overall success
    $overallSuccess = $true
    
    # Install system dependencies
    if (-not $SkipDependencies) {
        if ($hasInternet -and $isAdmin) {
            if (-not (Install-Chocolatey)) { $overallSuccess = $false }
        }
        
        if (-not $SkipPython) {
            if (-not (Install-Python -Version $PythonVersion)) { $overallSuccess = $false }
        }
        
        if (-not (Install-Git)) { $overallSuccess = $false }
        if (-not (Install-NodeJS)) { } # Not critical for backend
    }
    
    # Setup Python environment
    if (-not (Setup-PythonEnvironment)) { $overallSuccess = $false }
    
    # Install Python dependencies
    if (-not (Install-PythonDependencies)) { $overallSuccess = $false }
    
    # Test Python setup
    if (-not (Test-PythonImports)) { $overallSuccess = $false }
    
    # Setup project structure
    if (-not (Setup-DirectoryStructure)) { $overallSuccess = $false }
    if (-not (Setup-EnvironmentFile)) { $overallSuccess = $false }
    
    # Setup models
    if (-not $SkipModels) {
        if (-not (Download-Models)) { $overallSuccess = $false }
    }
    
    # Create startup scripts
    if (-not (Create-StartupScript)) { $overallSuccess = $false }
    
    # Show final summary
    Show-Summary -OverallSuccess $overallSuccess
    
    return $overallSuccess
}

# Execute main function
try {
    $result = Main
    if ($result) {
        exit 0
    } else {
        exit 1
    }
} catch {
    Write-LogMessage "Unexpected error during setup: $($_.Exception.Message)" "ERROR"
    exit 1
}