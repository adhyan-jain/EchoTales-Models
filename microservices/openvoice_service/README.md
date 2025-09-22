# EchoTales OpenVoice Microservice

A complete microservice solution for voice cloning and TTS in EchoTales, designed to handle OpenVoice dependency conflicts through service isolation.

## 🎯 Features

- **Voice Cloning**: Clone voices using your voice dataset with OpenVoice
- **Fallback TTS**: Automatic fallback to Edge TTS when OpenVoice is unavailable  
- **Voice Management**: Upload, list, and manage voice profiles
- **RESTful API**: Complete REST API with automatic documentation
- **Character Integration**: Smart voice selection based on character traits
- **Error Handling**: Comprehensive error handling with graceful degradation

## 🚀 Quick Start

### 1. Start the Microservice

```bash
# Option 1: Using the startup script (Windows)
cd microservices/openvoice_service
./start_service.bat

# Option 2: Manual startup
cd microservices/openvoice_service
pip install -r requirements.txt
python app.py
```

### 2. Verify Service is Running

Open your browser and go to:
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health
- **Service Status**: http://localhost:8001/

### 3. Test from Python

```python
from src.echotales.voice.openvoice_client import EchoTalesVoiceService

# Create service client
voice_service = EchoTalesVoiceService()

# Generate voice for a character
result = await voice_service.generate_character_voice(
    character_name="Princess Elena",
    text="Hello! Welcome to the kingdom of EchoTales!",
    character_traits={"gender": "female", "age": "young"},
    output_dir="output/audio"
)

print(f"Generated audio: {result['audio_file']}")
```

## 📡 API Endpoints

### Core Endpoints

- `GET /` - Service status and information
- `GET /health` - Health check with detailed status
- `GET /voices` - List all available voice profiles

### Voice Generation

- `POST /clone` - Clone a voice using reference audio
- `POST /tts` - Basic text-to-speech synthesis
- `GET /download/{filename}` - Download generated audio files

### Voice Management

- `POST /upload-reference` - Upload new reference voice samples
- `DELETE /voices/{voice_id}` - Delete custom voice profiles

## 🎭 Voice Profiles

The service automatically loads voice profiles from your dataset:

```
data/voice_dataset/
├── male_actors/
│   ├── Actor_01/
│   ├── Actor_02/
│   └── Actor_03/
└── female_actors/
    ├── Actor_01/
    ├── Actor_02/
    └── Actor_03/
```

Each profile includes:
- Gender classification
- Age range estimation  
- Voice characteristic analysis
- Sample audio files

## 🔧 Configuration

### Environment Variables

```bash
OPENVOICE_HOST=0.0.0.0      # Service host
OPENVOICE_PORT=8001         # Service port
OPENVOICE_DEBUG=True        # Debug mode
VOICE_DATASET_PATH=../../data/voice_dataset  # Path to voice dataset
```

### Service Settings

Edit `app.py` to customize:
- Model paths
- Audio output settings
- Timeout configurations
- Fallback voice mappings

## 🧪 Testing

### Run Comprehensive Tests

```bash
# From project root
python test_openvoice_microservice.py
```

### Manual Testing

```bash
# Test health
curl http://localhost:8001/health

# List voices
curl http://localhost:8001/voices

# Generate speech
curl -X POST http://localhost:8001/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello World", "voice_profile": "default"}'
```

## 🏗️ Architecture

### Microservice Benefits

1. **Isolation**: OpenVoice dependency conflicts are contained
2. **Scalability**: Service can be scaled independently
3. **Fallback**: Graceful degradation to Edge TTS
4. **Maintainability**: Easier to update and debug
5. **API First**: Clean REST interface for any client

### Client Integration

```python
# Main EchoTales app uses the client
from echotales.voice.openvoice_client import OpenVoiceClient

client = OpenVoiceClient("http://localhost:8001")
result = await client.clone_voice(text="Hello", reference_voice_id="male_Actor_01")
```

## 🐛 Troubleshooting

### Service Won't Start

1. **Check dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Check port availability**:
   ```bash
   netstat -an | findstr 8001
   ```

3. **Check logs**: Look for error messages in console output

### Voice Cloning Fails

1. **Verify voice dataset**: Ensure `data/voice_dataset/` exists with audio files
2. **Check OpenVoice**: Service will fallback to Edge TTS if OpenVoice fails
3. **Audio format**: Ensure voice samples are in supported formats (WAV, MP3, FLAC)

### No Audio Output

1. **Check file permissions**: Ensure service can write to output directory
2. **Verify downloads**: Use `/download/{filename}` endpoint to get audio files
3. **Format issues**: Service generates WAV files by default

## 📝 API Documentation

Once the service is running, visit http://localhost:8001/docs for interactive API documentation with:

- Request/response schemas
- Try-it-out functionality  
- Authentication details
- Error code explanations

## 🔄 Integration with Main App

The microservice integrates seamlessly with EchoTales:

```python
# In your main application
from echotales.voice.openvoice_client import generate_character_voice_sync

# Generate voice (sync version)
result = generate_character_voice_sync(
    character_name="Hero", 
    text="I shall save the kingdom!",
    character_traits={"gender": "male", "age": "adult"},
    output_dir="output/audio"
)
```

## 🚢 Deployment

### Development
- Run locally on port 8001
- Uses file-based storage
- Debug mode enabled

### Production  
- Deploy behind reverse proxy (nginx)
- Use environment variables for config
- Enable logging and monitoring
- Consider containerization with Docker

## 🔒 Security Notes

- Service binds to 0.0.0.0 by default (change for production)
- No authentication implemented (add as needed)
- File uploads are restricted to audio formats
- Rate limiting not implemented (add for production)

---

**EchoTales OpenVoice Service** - Making voice cloning simple and reliable! 🎙️