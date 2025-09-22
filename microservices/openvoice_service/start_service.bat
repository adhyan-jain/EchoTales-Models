@echo off
echo Starting EchoTales OpenVoice Microservice
echo =========================================

echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Checking OpenVoice setup...
if exist "../../models/voice_cloning/OpenVoice" (
    echo OpenVoice repository found
) else (
    echo Warning: OpenVoice repository not found, running in fallback mode
)

echo.
echo [3/3] Starting microservice...
echo Service will be available at: http://localhost:8001
echo API documentation at: http://localhost:8001/docs
echo.
echo Press Ctrl+C to stop the service
echo.

python app.py