#!/usr/bin/env python3
"""
Enhanced Edge TTS for EchoTales
Character-aware voice selection with gender, age, and personality mapping
"""

import asyncio
import json
import logging
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import edge_tts

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharacterVoiceSelector:
    """Selects appropriate voices based on character traits"""
    
    def __init__(self, voice_mappings_path: Optional[str] = None):
        self.voice_mappings_path = voice_mappings_path or self._get_default_mappings_path()
        self.voice_mappings = self._load_voice_mappings()
        self.available_voices = {}
        self._voice_cache = {}
        
    def _get_default_mappings_path(self) -> str:
        """Get the default path to voice mappings"""
        current_dir = Path(__file__).parent
        return str(current_dir / "voice_mappings.json")
    
    def _load_voice_mappings(self) -> Dict:
        """Load voice mappings from JSON file"""
        try:
            with open(self.voice_mappings_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load voice mappings: {e}")
            return self._get_default_mappings()
    
    def _get_default_mappings(self) -> Dict:
        """Fallback voice mappings if file loading fails"""
        return {
            "voice_categories": {
                "male": {
                    "young_adult": {
                        "friendly": ["en-US-BrandonNeural", "en-US-JacobNeural"],
                        "serious": ["en-US-ChristopherNeural", "en-US-EricNeural"]
                    }
                },
                "female": {
                    "young_adult": {
                        "friendly": ["en-US-AriaNeural", "en-US-JennyNeural"],
                        "serious": ["en-US-AshleyNeural", "en-US-CoraNeural"]
                    }
                }
            },
            "character_type_mappings": {
                "protagonist": {
                    "fallback_voices": {"male": "en-US-BrandonNeural", "female": "en-US-AriaNeural"}
                }
            }
        }
    
    async def initialize_available_voices(self):
        """Initialize the list of available Edge TTS voices"""
        try:
            voices = await edge_tts.list_voices()
            self.available_voices = {
                voice['ShortName']: {
                    'name': voice['ShortName'],
                    'display_name': voice['FriendlyName'],
                    'gender': voice['Gender'].lower(),
                    'locale': voice['Locale'],
                    'sample_rate': voice.get('SampleRateHertz', 24000),
                    'voice_type': voice.get('VoiceType', 'Neural')
                }
                for voice in voices if voice['Locale'].startswith('en')
            }
            logger.info(f"Loaded {len(self.available_voices)} English voices")
        except Exception as e:
            logger.error(f"Failed to load available voices: {e}")
            self.available_voices = {}
    
    def select_voice_for_character(
        self, 
        character_name: str,
        gender: str = "unknown",
        age_group: str = "young_adult",
        character_type: str = "protagonist",
        personality_traits: Optional[List[str]] = None
    ) -> str:
        """
        Select the most appropriate voice for a character
        
        Args:
            character_name: Name of the character
            gender: "male", "female", or "unknown"
            age_group: "child", "young_adult", "middle_aged", "elderly"
            character_type: "protagonist", "antagonist", "mentor", "child", "elder", "narrator"
            personality_traits: List of traits like ["friendly", "serious", "wise"]
        
        Returns:
            Edge TTS voice name
        """
        
        # Create a cache key for consistent voice assignment
        cache_key = f"{character_name}_{gender}_{age_group}_{character_type}"
        if cache_key in self._voice_cache:
            return self._voice_cache[cache_key]
        
        # Normalize inputs
        gender = gender.lower() if gender else "unknown"
        age_group = age_group.lower() if age_group else "young_adult"
        character_type = character_type.lower() if character_type else "protagonist"
        
        # If gender is unknown, infer from character name or default to neutral
        if gender == "unknown":
            gender = self._infer_gender_from_name(character_name)
        
        # Get personality traits
        if not personality_traits:
            personality_traits = self._get_default_traits_for_character_type(character_type)
        
        # Select voice based on criteria
        selected_voice = self._select_best_voice(gender, age_group, personality_traits, character_type)
        
        # Cache the selection
        self._voice_cache[cache_key] = selected_voice
        
        logger.info(f"Selected voice '{selected_voice}' for character '{character_name}' "
                   f"(gender: {gender}, age: {age_group}, type: {character_type})")
        
        return selected_voice
    
    def _infer_gender_from_name(self, character_name: str) -> str:
        """Simple gender inference from character name"""
        # This is a basic implementation - could be enhanced with a name database
        male_indicators = ['mr', 'sir', 'lord', 'king', 'prince', 'duke', 'baron']
        female_indicators = ['ms', 'mrs', 'miss', 'lady', 'queen', 'princess', 'duchess']
        
        name_lower = character_name.lower()
        
        for indicator in male_indicators:
            if indicator in name_lower:
                return "male"
        
        for indicator in female_indicators:
            if indicator in name_lower:
                return "female"
        
        # Default to male if uncertain (could be made configurable)
        return "male"
    
    def _get_default_traits_for_character_type(self, character_type: str) -> List[str]:
        """Get default personality traits for a character type"""
        type_mappings = self.voice_mappings.get("character_type_mappings", {})
        character_mapping = type_mappings.get(character_type, {})
        return character_mapping.get("default_traits", ["friendly"])
    
    def _select_best_voice(
        self, 
        gender: str, 
        age_group: str, 
        personality_traits: List[str],
        character_type: str
    ) -> str:
        """Select the best voice based on all criteria"""
        
        voice_categories = self.voice_mappings.get("voice_categories", {})
        
        # Get voices for gender and age group
        gender_voices = voice_categories.get(gender, {})
        age_voices = gender_voices.get(age_group, {})
        
        # Collect candidate voices based on personality traits
        candidate_voices = []
        for trait in personality_traits:
            trait_voices = age_voices.get(trait, [])
            candidate_voices.extend(trait_voices)
        
        # If no candidates found, try fallback for character type
        if not candidate_voices:
            type_mappings = self.voice_mappings.get("character_type_mappings", {})
            character_mapping = type_mappings.get(character_type, {})
            fallback_voices = character_mapping.get("fallback_voices", {})
            fallback_voice = fallback_voices.get(gender)
            
            if fallback_voice:
                candidate_voices = [fallback_voice]
        
        # If still no candidates, use any available voice for the gender
        if not candidate_voices:
            all_age_voices = []
            for age, traits in gender_voices.items():
                for trait, voices in traits.items():
                    all_age_voices.extend(voices)
            candidate_voices = all_age_voices
        
        # Final fallback - use any available voice
        if not candidate_voices:
            if gender == "male":
                candidate_voices = ["en-US-BrandonNeural"]
            else:
                candidate_voices = ["en-US-AriaNeural"]
        
        # Select randomly from candidates for variety, but use hash for consistency
        if candidate_voices:
            # Use character name hash to ensure same character always gets same voice
            hash_seed = hashlib.md5(f"{gender}_{age_group}_{character_type}".encode()).hexdigest()
            random.seed(hash_seed)
            selected_voice = random.choice(candidate_voices)
            random.seed()  # Reset seed
            return selected_voice
        
        return "en-US-AriaNeural"  # Ultimate fallback
    
    def get_voice_modifications(self, character_type: str, age_group: str) -> Dict[str, str]:
        """Get voice speed and pitch modifications"""
        modifications = {}
        
        voice_variations = self.voice_mappings.get("voice_variations", {})
        speed_modifiers = voice_variations.get("speed_modifiers", {})
        pitch_modifiers = voice_variations.get("pitch_modifiers", {})
        
        # Apply age-based modifications
        if age_group in speed_modifiers:
            modifications['speed'] = speed_modifiers[age_group]
        if age_group in pitch_modifiers:
            modifications['pitch'] = pitch_modifiers[age_group]
        
        # Apply character type modifications
        if character_type in speed_modifiers:
            modifications['speed'] = speed_modifiers[character_type]
        if character_type in pitch_modifiers:
            modifications['pitch'] = pitch_modifiers[character_type]
        
        return modifications


class EnhancedEdgeTTS:
    """Enhanced Edge TTS with character-aware voice selection"""
    
    def __init__(self, voice_mappings_path: Optional[str] = None):
        self.voice_selector = CharacterVoiceSelector(voice_mappings_path)
        self.initialized = False
        
    async def initialize(self):
        """Initialize the TTS system"""
        if not self.initialized:
            await self.voice_selector.initialize_available_voices()
            self.initialized = True
            logger.info("Enhanced Edge TTS initialized")
    
    async def synthesize_character_speech(
        self,
        text: str,
        character_name: str,
        output_path: str,
        gender: str = "unknown",
        age_group: str = "young_adult",
        character_type: str = "protagonist",
        personality_traits: Optional[List[str]] = None,
        apply_modifications: bool = True
    ) -> bool:
        """
        Synthesize speech for a specific character
        
        Args:
            text: Text to synthesize
            character_name: Name of the character speaking
            output_path: Path to save the audio file
            gender: Character's gender
            age_group: Character's age group
            character_type: Type of character (protagonist, antagonist, etc.)
            personality_traits: List of personality traits
            apply_modifications: Whether to apply voice modifications
        
        Returns:
            True if successful, False otherwise
        """
        
        try:
            await self.initialize()
            
            # Select appropriate voice
            voice_name = self.voice_selector.select_voice_for_character(
                character_name=character_name,
                gender=gender,
                age_group=age_group,
                character_type=character_type,
                personality_traits=personality_traits
            )
            
            # Get voice modifications
            modifications = {}
            if apply_modifications:
                modifications = self.voice_selector.get_voice_modifications(
                    character_type, age_group
                )
            
            # Apply SSML modifications if needed
            final_text = self._apply_ssml_modifications(text, modifications)
            
            # Generate speech
            communicate = edge_tts.Communicate(text=final_text, voice=voice_name)
            await communicate.save(output_path)
            
            output_file = Path(output_path)
            if output_file.exists():
                logger.info(f"Generated speech for '{character_name}' using voice '{voice_name}': {output_path}")
                return True
            else:
                logger.error(f"Failed to generate speech file: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Speech synthesis failed for character '{character_name}': {e}")
            return False
    
    def _apply_ssml_modifications(self, text: str, modifications: Dict[str, str]) -> str:
        """Apply SSML modifications to text"""
        if not modifications:
            return text
        
        ssml_parts = []
        
        # Add speed modification
        if 'speed' in modifications:
            speed_value = modifications['speed']
            ssml_parts.append(f'<prosody rate="{speed_value}">')
        
        # Add pitch modification
        if 'pitch' in modifications:
            pitch_value = modifications['pitch']
            if ssml_parts:
                # Combine with existing prosody tag
                ssml_parts[-1] = ssml_parts[-1].replace('>', f' pitch="{pitch_value}">')
            else:
                ssml_parts.append(f'<prosody pitch="{pitch_value}">')
        
        # Add text
        ssml_parts.append(text)
        
        # Close tags
        if 'speed' in modifications or 'pitch' in modifications:
            ssml_parts.append('</prosody>')
        
        return ''.join(ssml_parts)
    
    async def list_character_voices(self) -> Dict[str, List[str]]:
        """List available voices by category"""
        await self.initialize()
        
        result = {}
        voice_categories = self.voice_selector.voice_mappings.get("voice_categories", {})
        
        for gender, age_groups in voice_categories.items():
            result[gender] = []
            for age_group, traits in age_groups.items():
                for trait, voices in traits.items():
                    result[gender].extend(voices)
            # Remove duplicates
            result[gender] = list(set(result[gender]))
        
        return result
    
    async def test_character_voices(self, test_text: str = "Hello, this is a test of character voice selection."):
        """Test different character voice selections"""
        await self.initialize()
        
        test_characters = [
            ("Hero", "male", "young_adult", "protagonist", ["friendly", "confident"]),
            ("Villain", "male", "middle_aged", "antagonist", ["serious", "authoritative"]),
            ("Wise_Elder", "female", "elderly", "mentor", ["wise", "warm"]),
            ("Child", "female", "child", "child", ["playful", "innocent"]),
            ("Narrator", "male", "middle_aged", "narrator", ["warm", "authoritative"])
        ]
        
        for char_name, gender, age, char_type, traits in test_characters:
            output_path = f"test_{char_name.lower()}_voice.wav"
            success = await self.synthesize_character_speech(
                text=f"{test_text} I am {char_name}.",
                character_name=char_name,
                output_path=output_path,
                gender=gender,
                age_group=age,
                character_type=char_type,
                personality_traits=traits
            )
            
            if success:
                print(f"✅ Generated test audio for {char_name}: {output_path}")
            else:
                print(f"❌ Failed to generate audio for {char_name}")


async def main():
    """Test the Enhanced Edge TTS system"""
    print("Enhanced Edge TTS for EchoTales")
    print("=" * 50)
    
    # Initialize TTS system
    tts = EnhancedEdgeTTS()
    await tts.initialize()
    
    # Test voice selection
    print("\n🎭 Testing Character Voice Selection")
    print("-" * 30)
    
    # Test different character types
    await tts.test_character_voices()
    
    # List available voices
    print("\n🗣️ Available Voices by Category")
    print("-" * 30)
    
    voice_categories = await tts.list_character_voices()
    for gender, voices in voice_categories.items():
        print(f"\n{gender.upper()} voices ({len(voices)} total):")
        for voice in voices[:5]:  # Show first 5
            print(f"  - {voice}")
        if len(voices) > 5:
            print(f"  ... and {len(voices) - 5} more")
    
    print(f"\n✅ Enhanced Edge TTS system is ready!")
    print(f"🎬 Generated test audio files for different character types")


if __name__ == "__main__":
    asyncio.run(main())