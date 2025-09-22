"""
EchoTales TTS Module
Enhanced text-to-speech with character-aware voice selection
"""

from .enhanced_edge_tts import EnhancedEdgeTTS, CharacterVoiceSelector
from .character_voice_processor import CharacterVoiceProcessor

__all__ = ['EnhancedEdgeTTS', 'CharacterVoiceSelector', 'CharacterVoiceProcessor']
