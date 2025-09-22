# BookNLP API Service Startup Script for PowerShell

Write-Host "Starting BookNLP API Service..." -ForegroundColor Green

# Check if we're in the right directory
if (!(Test-Path "app.py")) {
    Write-Host "Error: app.py not found. Please run this script from the booknlp_service directory." -ForegroundColor Red
    exit 1
}

# Check if Python is available
try {
    python --version | Out-Null
    Write-Host "✓ Python is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not available. Please install Python first." -ForegroundColor Red
    exit 1
}

# Install dependencies if requirements.txt exists
if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install dependencies." -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
}

# Start the service
Write-Host "Starting BookNLP API on http://localhost:8002" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the service" -ForegroundColor Yellow
Write-Host "API Documentation available at: http://localhost:8002/docs" -ForegroundColor Cyan

python app.py