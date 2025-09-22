@echo off
echo EchoTales Enhanced - Quick Setup Script
echo =====================================

echo.
echo [1/4] Installing minimal dependencies...
pip install flask flask-cors python-dotenv pyyaml requests

echo.
echo [2/4] Creating directories...
if not exist "data\cache" mkdir "data\cache"
if not exist "data\output\images" mkdir "data\output\images"
if not exist "data\output\audio" mkdir "data\output\audio"
if not exist "data\voice_dataset" mkdir "data\voice_dataset"
if not exist "logs" mkdir "logs"
if not exist "models" mkdir "models"
if not exist "configs" mkdir "configs"

echo.
echo [3/4] Creating .env file...
if not exist ".env" (
    echo # Minimal EchoTales Environment Configuration> .env
    echo FLASK_ENV=development>> .env
    echo FLASK_DEBUG=True>> .env
    echo FLASK_HOST=127.0.0.1>> .env
    echo FLASK_PORT=5000>> .env
    echo LOG_LEVEL=INFO>> .env
    echo SECRET_KEY=dev-secret-key-change-in-production>> .env
    echo CORS_ORIGINS=http://localhost:3000,http://localhost:3001>> .env
    echo.>> .env
    echo # Optional API keys (leave empty for mock mode^)>> .env
    echo OPENAI_API_KEY=>> .env
    echo GEMINI_API_KEY=>> .env
    echo STABILITY_API_KEY=>> .env
    echo HUGGINGFACE_API_TOKEN=>> .env
    echo.>> .env
    echo # Mock mode settings>> .env
    echo ENABLE_MOCK_APIS=True>> .env
    echo MOCK_RESPONSE_DELAY=1>> .env
)

echo.
echo [4/4] Setup complete!
echo.
echo To start the development server:
echo   python run_dev.py
echo.
echo The server will be available at: http://127.0.0.1:5000
echo.
echo Available endpoints:
echo   - GET  /health
echo   - GET  /api/status  
echo   - POST /api/novels/process
echo   - POST /api/images/generate
echo.
pause