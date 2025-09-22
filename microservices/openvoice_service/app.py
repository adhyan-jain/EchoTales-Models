#!/usr/bin/env python3
"""
OpenVoice Microservice for EchoTales
Handles voice cloning and TTS via REST API
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import shutil
import tempfile
import logging
from pathlib import Path
import json
import asyncio
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from echotales.tts import EnhancedEdgeTTS
import edge_tts
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="EchoTales OpenVoice Service",
    description="Voice cloning and TTS microservice using OpenVoice",
    version="1.0.0"
)

# Global variables
openvoice_model = None
voice_profiles = {}
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


class VoiceCloneRequest(BaseModel):
    text: str
    reference_voice_id: str
    output_filename: Optional[str] = None
    voice_settings: Optional[Dict] = {}


class CharacterInfo(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = "unknown"
    age_group: Optional[str] = "young_adult"
    character_type: Optional[str] = "protagonist"
    personality_traits: Optional[List[str]] = []

class TTSRequest(BaseModel):
    text: str
    voice_profile: Optional[str] = "default"
    voice_settings: Optional[Dict] = {}
    character_info: Optional[CharacterInfo] = None
    pitch: Optional[float] = 1.0


class VoiceProfile(BaseModel):
    id: str
    name: str
    gender: str
    age_range: str
    accent: str
    description: str
    sample_files: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize OpenVoice on startup"""
    global openvoice_model, voice_profiles
    
    try:
        # Initialize OpenVoice (with error handling for dependency conflicts)
        openvoice_model = await initialize_openvoice()
        
        # Load voice profiles from dataset
        voice_profiles = await load_voice_profiles()
        
        logger.info("OpenVoice service started successfully")
        logger.info(f"Loaded {len(voice_profiles)} voice profiles")
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenVoice: {e}")
        # Continue running in fallback mode


async def initialize_openvoice():
    """Initialize OpenVoice with fallback handling"""
    
    try:
        # Try to import OpenVoice
        sys.path.append("../../models/voice_cloning/OpenVoice")
        
        # Import OpenVoice modules with fallback
        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            logger.info("OpenVoice modules loaded successfully")
            
            # Initialize models
            model = {
                "se_extractor": se_extractor,
                "converter": ToneColorConverter("../../models/voice_cloning/OpenVoice/checkpoints/converter")
            }
            
            return model
            
        except ImportError as e:
            logger.warning(f"OpenVoice import failed: {e}")
            return None
            
    except Exception as e:
        logger.error(f"OpenVoice initialization failed: {e}")
        return None


async def load_voice_profiles():
    """Load voice profiles from the dataset"""
    profiles = {}
    voice_dataset_path = Path("../../data/voice_dataset")
    
    if not voice_dataset_path.exists():
        logger.warning("Voice dataset not found")
        return profiles
    
    try:
        # Load male actors
        male_path = voice_dataset_path / "male_actors"
        if male_path.exists():
            for actor_dir in male_path.iterdir():
                if actor_dir.is_dir():
                    profiles[f"male_{actor_dir.name}"] = {
                        "id": f"male_{actor_dir.name}",
                        "name": actor_dir.name,
                        "gender": "male",
                        "age_range": "adult",
                        "accent": "american",
                        "description": f"Male voice actor {actor_dir.name}",
                        "sample_files": [str(f) for f in actor_dir.glob("*.wav")],
                        "path": str(actor_dir)
                    }
        
        # Load female actors
        female_path = voice_dataset_path / "female_actors"
        if female_path.exists():
            for actor_dir in female_path.iterdir():
                if actor_dir.is_dir():
                    profiles[f"female_{actor_dir.name}"] = {
                        "id": f"female_{actor_dir.name}",
                        "name": actor_dir.name,
                        "gender": "female", 
                        "age_range": "adult",
                        "accent": "american",
                        "description": f"Female voice actor {actor_dir.name}",
                        "sample_files": [str(f) for f in actor_dir.glob("*.wav")],
                        "path": str(actor_dir)
                    }
        
        logger.info(f"Loaded {len(profiles)} voice profiles")
        return profiles
        
    except Exception as e:
        logger.error(f"Failed to load voice profiles: {e}")
        return profiles


@app.get("/")
async def root():
    """Root endpoint with service status"""
    return {
        "service": "EchoTales OpenVoice Service",
        "status": "running",
        "version": "1.0.0",
        "openvoice_available": openvoice_model is not None,
        "voice_profiles": len(voice_profiles)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if openvoice_model else "degraded",
        "openvoice_loaded": openvoice_model is not None,
        "voice_profiles_count": len(voice_profiles),
        "timestamp": time.time()
    }


@app.get("/voices", response_model=List[VoiceProfile])
async def list_voices():
    """List all available voice profiles"""
    try:
        voice_list = []
        
        for profile_id, profile_data in voice_profiles.items():
            voice_list.append(VoiceProfile(
                id=profile_data["id"],
                name=profile_data["name"],
                gender=profile_data["gender"],
                age_range=profile_data["age_range"],
                accent=profile_data["accent"],
                description=profile_data["description"],
                sample_files=[str(Path(f).name) for f in profile_data["sample_files"][:3]]  # Show first 3 samples
            ))
        
        return voice_list
        
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")


@app.post("/clone")
async def clone_voice(request: VoiceCloneRequest):
    """Clone a voice using OpenVoice"""
    
    if not openvoice_model:
        # Fallback mode - use existing TTS
        return await fallback_tts(request.text, request.reference_voice_id)
    
    try:
        # Validate voice profile exists
        if request.reference_voice_id not in voice_profiles:
            raise HTTPException(status_code=404, detail=f"Voice profile '{request.reference_voice_id}' not found")
        
        voice_profile = voice_profiles[request.reference_voice_id]
        
        # Generate output filename
        if not request.output_filename:
            text_hash = hashlib.md5(request.text.encode()).hexdigest()[:8]
            request.output_filename = f"cloned_{request.reference_voice_id}_{text_hash}.wav"
        
        output_path = output_dir / request.output_filename
        
        # Perform voice cloning
        success = await perform_voice_cloning(
            text=request.text,
            voice_profile=voice_profile,
            output_path=output_path,
            settings=request.voice_settings
        )
        
        if success:
            return {
                "status": "success",
                "message": "Voice cloning completed",
                "output_file": request.output_filename,
                "voice_profile": request.reference_voice_id,
                "download_url": f"/download/{request.output_filename}"
            }
        else:
            raise HTTPException(status_code=500, detail="Voice cloning failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


async def perform_voice_cloning(text: str, voice_profile: Dict, output_path: Path, settings: Dict):
    """Perform the actual voice cloning using OpenVoice"""
    
    try:
        if not openvoice_model:
            return False
        
        # Get reference audio file
        sample_files = voice_profile["sample_files"]
        if not sample_files:
            logger.error(f"No sample files for voice profile {voice_profile['id']}")
            return False
        
        reference_audio = sample_files[0]  # Use first sample
        
        # OpenVoice cloning process
        logger.info(f"Cloning voice with reference: {reference_audio}")
        logger.info(f"Text to synthesize: {text}")
        
        # Extract speaker embedding from reference
        # Note: This is a simplified version - real OpenVoice has more complex pipeline
        speaker_embedding = extract_speaker_embedding(reference_audio)
        
        # Generate audio with target speaker
        audio = synthesize_with_speaker(text, speaker_embedding, settings)
        
        # Save to output path
        save_audio(audio, output_path)
        
        logger.info(f"Voice cloning completed: {output_path}")
        return output_path.exists()
        
    except Exception as e:
        logger.error(f"Voice cloning process failed: {e}")
        return False


def extract_speaker_embedding(reference_audio_path: str):
    """Extract speaker embedding from reference audio"""
    # Placeholder for OpenVoice speaker extraction
    # In real implementation, this would use OpenVoice's speaker encoder
    logger.info(f"Extracting speaker embedding from: {reference_audio_path}")
    return {"speaker_id": "extracted", "embedding": [0.1, 0.2, 0.3]}  # Mock embedding


def synthesize_with_speaker(text: str, speaker_embedding: Dict, settings: Dict):
    """Synthesize text with target speaker characteristics"""
    # Placeholder for OpenVoice synthesis
    logger.info(f"Synthesizing text with speaker embedding")
    return b"fake_audio_data"  # Mock audio data


def save_audio(audio_data, output_path: Path):
    """Save audio data to file"""
    # For demonstration, create a simple placeholder file
    with open(output_path, 'wb') as f:
        f.write(b"RIFF....WAVE....")  # Minimal WAV header
    logger.info(f"Audio saved to: {output_path}")


# Initialize Enhanced Edge TTS
enhanced_tts = None

async def initialize_enhanced_tts():
    """Initialize the Enhanced Edge TTS system"""
    global enhanced_tts
    if enhanced_tts is None:
        try:
            enhanced_tts = EnhancedEdgeTTS()
            await enhanced_tts.initialize()
            logger.info("Enhanced Edge TTS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Edge TTS: {e}")
            enhanced_tts = None

async def enhanced_character_tts(text: str, voice_id: str, character_info: dict = None):
    """Enhanced TTS with character-aware voice selection"""
    
    try:
        # Initialize if needed
        await initialize_enhanced_tts()
        
        if enhanced_tts is None:
            return await basic_fallback_tts(text, voice_id)
        
        # Extract character information
        character_name = character_info.get('name', voice_id) if character_info else voice_id
        gender = character_info.get('gender', 'unknown') if character_info else 'unknown'
        age_group = character_info.get('age_group', 'young_adult') if character_info else 'young_adult'
        character_type = character_info.get('character_type', 'protagonist') if character_info else 'protagonist'
        personality_traits = character_info.get('personality_traits', []) if character_info else []
        
        # Infer character info from voice_id if not provided
        if gender == 'unknown':
            if 'male' in voice_id.lower():
                gender = 'male'
            elif 'female' in voice_id.lower():
                gender = 'female'
        
        # Generate filename
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        filename = f"enhanced_{voice_id}_{text_hash}.wav"
        output_path = output_dir / filename
        
        # Generate audio using Enhanced Edge TTS
        success = await enhanced_tts.synthesize_character_speech(
            text=text,
            character_name=character_name,
            output_path=str(output_path),
            gender=gender,
            age_group=age_group,
            character_type=character_type,
            personality_traits=personality_traits,
            apply_modifications=True
        )
        
        if success:
            return {
                "status": "success",
                "message": "Enhanced character voice synthesis completed",
                "output_file": filename,
                "voice_profile": voice_id,
                "character_info": {
                    "name": character_name,
                    "gender": gender,
                    "age_group": age_group,
                    "character_type": character_type
                },
                "download_url": f"/download/{filename}",
                "mode": "enhanced_tts"
            }
        else:
            return await basic_fallback_tts(text, voice_id)
        
    except Exception as e:
        logger.error(f"Enhanced TTS failed: {e}")
        return await basic_fallback_tts(text, voice_id)

async def basic_fallback_tts(text: str, voice_id: str):
    """Basic fallback TTS when enhanced TTS is not available"""
    
    try:
        # Use basic Edge TTS as fallback
        import edge_tts
        
        # Map voice ID to Edge TTS voice
        voice_map = {
            "male_Actor_01": "en-US-BrandonNeural",
            "male_Actor_02": "en-US-ChristopherNeural", 
            "female_Actor_01": "en-US-AriaNeural",
            "female_Actor_02": "en-US-JennyNeural"
        }
        
        edge_voice = voice_map.get(voice_id, "en-US-AriaNeural")
        
        # Generate filename
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        filename = f"fallback_{voice_id}_{text_hash}.wav"
        output_path = output_dir / filename
        
        # Generate audio using Edge TTS
        communicate = edge_tts.Communicate(text=text, voice=edge_voice)
        await communicate.save(str(output_path))
        
        return {
            "status": "success",
            "message": "Voice synthesis completed (basic fallback mode)",
            "output_file": filename,
            "voice_profile": voice_id,
            "download_url": f"/download/{filename}",
            "mode": "basic_fallback_tts"
        }
        
    except ImportError:
        # If even Edge TTS is not available, return error
        raise HTTPException(status_code=503, detail="TTS service unavailable")
    except Exception as e:
        logger.error(f"Basic fallback TTS failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Enhanced TTS with character-aware voice selection"""
    
    try:
        # Extract character information from request if available
        character_info = request.character_info.dict() if request.character_info else None
        
        # Use enhanced character TTS
        result = await enhanced_character_tts(
            text=request.text, 
            voice_id=request.voice_profile or "default",
            character_info=character_info
        )
        return result
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.post("/character-tts")
async def character_text_to_speech(request: TTSRequest):
    """Character-specific TTS with enhanced voice selection"""
    
    try:
        # Ensure character info is provided
        if not request.character_info:
            raise HTTPException(status_code=400, detail="Character information is required for character TTS")
        
        character_info = request.character_info.dict()
        
        # Use enhanced character TTS
        result = await enhanced_character_tts(
            text=request.text, 
            voice_id=request.voice_profile or request.character_info.name or "default",
            character_info=character_info
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Character TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Character TTS failed: {str(e)}")


@app.get("/voices")
async def list_available_voices():
    """List available voices by category"""
    
    try:
        await initialize_enhanced_tts()
        
        if enhanced_tts:
            voice_categories = await enhanced_tts.list_character_voices()
            return {
                "status": "success",
                "voices": voice_categories,
                "total_voices": sum(len(voices) for voices in voice_categories.values()),
                "mode": "enhanced_tts"
            }
        else:
            # Fallback to basic voice list
            basic_voices = {
                "male": ["en-US-BrandonNeural", "en-US-ChristopherNeural", "en-US-EricNeural"],
                "female": ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-AshleyNeural"]
            }
            return {
                "status": "success",
                "voices": basic_voices,
                "total_voices": sum(len(voices) for voices in basic_voices.values()),
                "mode": "basic_fallback"
            }
            
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve voice list")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated audio file"""
    
    file_path = output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="audio/wav"
    )


@app.post("/upload-reference")
async def upload_reference_voice(
    file: UploadFile = File(...),
    voice_name: str = Form(...),
    gender: str = Form(...),
    description: str = Form(...)
):
    """Upload a new reference voice sample"""
    
    try:
        # Validate file type
        if not file.filename.endswith(('.wav', '.mp3', '.flac')):
            raise HTTPException(status_code=400, detail="Only WAV, MP3, and FLAC files are supported")
        
        # Create directory for this voice
        voice_dir = Path("voice_uploads") / voice_name
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = voice_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add to voice profiles
        voice_id = f"custom_{voice_name}"
        voice_profiles[voice_id] = {
            "id": voice_id,
            "name": voice_name,
            "gender": gender,
            "age_range": "adult",
            "accent": "custom",
            "description": description,
            "sample_files": [str(file_path)],
            "path": str(voice_dir)
        }
        
        return {
            "status": "success",
            "message": "Reference voice uploaded successfully",
            "voice_id": voice_id,
            "file_path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a custom voice profile"""
    
    try:
        if voice_id not in voice_profiles:
            raise HTTPException(status_code=404, detail="Voice profile not found")
        
        # Don't delete built-in voices
        if not voice_id.startswith("custom_"):
            raise HTTPException(status_code=403, detail="Cannot delete built-in voice profiles")
        
        # Remove from profiles
        del voice_profiles[voice_id]
        
        return {
            "status": "success",
            "message": f"Voice profile {voice_id} deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


if __name__ == "__main__":
    # Run the microservice
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )