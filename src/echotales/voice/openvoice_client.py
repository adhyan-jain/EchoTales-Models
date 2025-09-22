"""
OpenVoice Client for EchoTales
Communicates with the OpenVoice microservice
"""

import httpx
import asyncio
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class OpenVoiceClient:
    """Client for communicating with OpenVoice microservice"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        self.timeout = httpx.Timeout(60.0)  # 60 seconds timeout for voice generation
        
    async def health_check(self) -> Dict[str, Any]:
        """Check if OpenVoice service is healthy"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            logger.error(f"OpenVoice service connection failed: {e}")
            return {"status": "unavailable", "error": str(e)}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def list_voices(self) -> List[Dict[str, Any]]:
        """List all available voice profiles"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/voices")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []
    
    async def clone_voice(self, 
                         text: str, 
                         reference_voice_id: str,
                         output_filename: Optional[str] = None,
                         voice_settings: Optional[Dict] = None) -> Dict[str, Any]:
        """Clone a voice using OpenVoice"""
        
        try:
            request_data = {
                "text": text,
                "reference_voice_id": reference_voice_id,
                "voice_settings": voice_settings or {}
            }
            
            if output_filename:
                request_data["output_filename"] = output_filename
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/clone",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Voice cloning successful: {result.get('output_file')}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Voice cloning HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                "status": "error", 
                "error": f"HTTP {e.response.status_code}: {e.response.text}"
            }
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def synthesize_speech(self, 
                              text: str,
                              voice_profile: str = "default",
                              speed: float = 1.0,
                              pitch: float = 1.0) -> Dict[str, Any]:
        """Basic TTS synthesis"""
        
        try:
            request_data = {
                "text": text,
                "voice_profile": voice_profile,
                "speed": speed,
                "pitch": pitch
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/tts",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Speech synthesis successful: {result.get('output_file')}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Speech synthesis HTTP error: {e.response.status_code}")
            return {
                "status": "error", 
                "error": f"HTTP {e.response.status_code}: {e.response.text}"
            }
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def download_audio(self, filename: str, local_path: str) -> bool:
        """Download generated audio file"""
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/download/{filename}")
                response.raise_for_status()
                
                # Save to local file
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Audio downloaded to: {local_path}")
                return True
                
        except Exception as e:
            logger.error(f"Audio download failed: {e}")
            return False
    
    async def upload_reference_voice(self, 
                                   file_path: str,
                                   voice_name: str,
                                   gender: str,
                                   description: str) -> Dict[str, Any]:
        """Upload a new reference voice sample"""
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {"status": "error", "error": "File not found"}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(file_path, 'rb') as f:
                    files = {"file": (file_path.name, f, "audio/wav")}
                    data = {
                        "voice_name": voice_name,
                        "gender": gender,
                        "description": description
                    }
                    
                    response = await client.post(
                        f"{self.base_url}/upload-reference",
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    return response.json()
                    
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def delete_voice(self, voice_id: str) -> Dict[str, Any]:
        """Delete a custom voice profile"""
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(f"{self.base_url}/voices/{voice_id}")
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Voice deletion failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def is_available(self) -> bool:
        """Synchronously check if service is available"""
        try:
            result = asyncio.run(self.health_check())
            return result.get("status") in ["healthy", "degraded"]
        except:
            return False


class EchoTalesVoiceService:
    """High-level voice service for EchoTales"""
    
    def __init__(self, openvoice_url: str = "http://localhost:8001"):
        self.openvoice_client = OpenVoiceClient(openvoice_url)
        self.fallback_voices = {
            "male_young": "en-US-BrandonNeural",
            "male_middle": "en-US-ChristopherNeural",
            "male_old": "en-US-GuyNeural",
            "female_young": "en-US-AriaNeural",
            "female_middle": "en-US-JennyNeural", 
            "female_old": "en-US-NancyNeural"
        }
    
    async def generate_character_voice(self, 
                                     character_name: str,
                                     text: str,
                                     character_traits: Dict[str, Any],
                                     output_dir: str) -> Dict[str, Any]:
        """Generate voice for a character based on their traits"""
        
        try:
            # Check if OpenVoice service is available
            health = await self.openvoice_client.health_check()
            
            if health.get("status") == "healthy":
                # Use OpenVoice service
                voice_id = self._select_voice_profile(character_traits)
                
                output_filename = f"{character_name.replace(' ', '_')}_voice.wav"
                
                result = await self.openvoice_client.clone_voice(
                    text=text,
                    reference_voice_id=voice_id,
                    output_filename=output_filename
                )
                
                if result.get("status") == "success":
                    # Download the generated audio
                    local_path = Path(output_dir) / output_filename
                    download_success = await self.openvoice_client.download_audio(
                        result["output_file"], 
                        str(local_path)
                    )
                    
                    if download_success:
                        return {
                            "status": "success",
                            "audio_file": str(local_path),
                            "character": character_name,
                            "voice_profile": voice_id,
                            "service": "openvoice"
                        }
            
            # Fallback to Edge TTS
            return await self._fallback_tts(character_name, text, character_traits, output_dir)
            
        except Exception as e:
            logger.error(f"Character voice generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _select_voice_profile(self, character_traits: Dict[str, Any]) -> str:
        """Select appropriate voice profile based on character traits"""
        
        # Extract traits
        gender = character_traits.get("gender", "unknown").lower()
        age = character_traits.get("age", "adult").lower()
        
        # Simple mapping logic
        if gender == "male":
            if "young" in age or "teen" in age:
                return "male_Actor_01"
            elif "old" in age or "elder" in age:
                return "male_Actor_03" 
            else:
                return "male_Actor_02"
        elif gender == "female":
            if "young" in age or "teen" in age:
                return "female_Actor_01"
            elif "old" in age or "elder" in age:
                return "female_Actor_03"
            else:
                return "female_Actor_02"
        else:
            # Default to neutral voice
            return "female_Actor_01"
    
    async def _fallback_tts(self, 
                          character_name: str,
                          text: str, 
                          character_traits: Dict[str, Any],
                          output_dir: str) -> Dict[str, Any]:
        """Fallback TTS using Edge TTS"""
        
        try:
            import edge_tts
            
            # Select Edge TTS voice
            voice_key = self._get_fallback_voice_key(character_traits)
            edge_voice = self.fallback_voices.get(voice_key, "en-US-AriaNeural")
            
            # Generate audio
            output_filename = f"{character_name.replace(' ', '_')}_voice.wav"
            output_path = Path(output_dir) / output_filename
            
            communicate = edge_tts.Communicate(text=text, voice=edge_voice)
            await communicate.save(str(output_path))
            
            return {
                "status": "success",
                "audio_file": str(output_path),
                "character": character_name,
                "voice_profile": edge_voice,
                "service": "edge_tts_fallback"
            }
            
        except ImportError:
            return {"status": "error", "error": "No TTS service available"}
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_fallback_voice_key(self, character_traits: Dict[str, Any]) -> str:
        """Get fallback voice key based on character traits"""
        
        gender = character_traits.get("gender", "female").lower()
        age = character_traits.get("age", "adult").lower()
        
        if gender == "male":
            if "young" in age:
                return "male_young"
            elif "old" in age:
                return "male_old"
            else:
                return "male_middle"
        else:
            if "young" in age:
                return "female_young"
            elif "old" in age:
                return "female_old"
            else:
                return "female_middle"
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get all available voice profiles"""
        
        voices = []
        
        # Try to get OpenVoice profiles
        try:
            openvoice_voices = await self.openvoice_client.list_voices()
            voices.extend(openvoice_voices)
        except:
            pass
        
        # Add fallback voices
        for key, edge_voice in self.fallback_voices.items():
            voices.append({
                "id": f"fallback_{key}",
                "name": key.replace("_", " ").title(),
                "gender": key.split("_")[0],
                "age_range": key.split("_")[1],
                "accent": "american",
                "description": f"Edge TTS {key.replace('_', ' ')} voice",
                "service": "edge_tts"
            })
        
        return voices


# Convenience functions for easy use
async def generate_character_voice(character_name: str, 
                                 text: str,
                                 character_traits: Dict[str, Any],
                                 output_dir: str = "data/output/audio") -> Dict[str, Any]:
    """Generate voice for a character"""
    
    service = EchoTalesVoiceService()
    return await service.generate_character_voice(character_name, text, character_traits, output_dir)


async def list_available_voices() -> List[Dict[str, Any]]:
    """List all available voices"""
    
    service = EchoTalesVoiceService()
    return await service.get_available_voices()


# Sync wrapper for convenience
def generate_character_voice_sync(character_name: str, 
                                text: str,
                                character_traits: Dict[str, Any],
                                output_dir: str = "data/output/audio") -> Dict[str, Any]:
    """Synchronous version of character voice generation"""
    
    return asyncio.run(generate_character_voice(character_name, text, character_traits, output_dir))


if __name__ == "__main__":
    # Test the client
    async def test_client():
        client = OpenVoiceClient()
        
        # Check health
        health = await client.health_check()
        print(f"Service health: {health}")
        
        # List voices
        voices = await client.list_voices()
        print(f"Available voices: {len(voices)}")
        
        # Test voice generation
        if voices:
            result = await client.clone_voice(
                text="Hello, this is a test of the OpenVoice microservice integration.",
                reference_voice_id=voices[0]["id"]
            )
            print(f"Voice generation result: {result}")
    
    asyncio.run(test_client())