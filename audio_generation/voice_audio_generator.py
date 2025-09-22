#!/usr/bin/env python3
"""
EchoTales Voice Audio Generator
Generates audio files from JSON character and dialogue data using Microsoft Edge TTS
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import edge_tts
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_audio_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceAudioGenerator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.json_dir = self.data_dir / "output" / "json"
        self.audio_dir = self.data_dir / "output" / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Microsoft Edge TTS voice mappings
        self.edge_voices = {
            'male': [
                'en-US-AriaNeural',
                'en-US-DavisNeural', 
                'en-US-GuyNeural',
                'en-US-JasonNeural',
                'en-US-TonyNeural',
                'en-GB-RyanNeural',
                'en-GB-ThomasNeural',
                'en-AU-WilliamNeural'
            ],
            'female': [
                'en-US-AriaNeural',
                'en-US-JennyNeural',
                'en-US-MichelleNeural',
                'en-US-MonicaNeural',
                'en-US-SaraNeural',
                'en-GB-SoniaNeural',
                'en-GB-LibbyNeural',
                'en-AU-NatashaNeural'
            ],
            'narrator': 'en-US-GuyNeural'  # Deep, authoritative voice for narrator
        }
        
        self.character_voice_mapping = {}
        self.characters = {}
        self.chapters = {}
        self.dialogue = {}
        self.voice_config = {}
        
    def load_json_data(self):
        """Load all JSON data files"""
        try:
            # Load characters
            characters_file = self.json_dir / "characters.json"
            if characters_file.exists():
                with open(characters_file, 'r', encoding='utf-8') as f:
                    characters_data = json.load(f)
                    self.characters = {char['character_id']: char for char in characters_data['characters']}
                logger.info(f"Loaded {len(self.characters)} characters")
            
            # Load chapters
            chapters_file = self.json_dir / "chapters.json"
            if chapters_file.exists():
                with open(chapters_file, 'r', encoding='utf-8') as f:
                    chapters_data = json.load(f)
                    self.chapters = {chapter['chapter_id']: chapter for chapter in chapters_data['chapters']}
                logger.info(f"Loaded {len(self.chapters)} chapters")
            
            # Load dialogue
            dialogue_file = self.json_dir / "dialogue.json"
            if dialogue_file.exists():
                with open(dialogue_file, 'r', encoding='utf-8') as f:
                    dialogue_data = json.load(f)
                    self.dialogue = {dlg['dialogue_id']: dlg for dlg in dialogue_data['dialogue']}
                logger.info(f"Loaded {len(self.dialogue)} dialogue lines")
            
            # Load voice config
            voice_config_file = self.json_dir / "voice_config.json"
            if voice_config_file.exists():
                with open(voice_config_file, 'r', encoding='utf-8') as f:
                    self.voice_config = json.load(f)
                logger.info(f"Loaded voice configuration with {len(self.voice_config.get('voice_profiles', {}))} profiles")
                    
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            raise

    def assign_voices_to_characters(self):
        """Assign Edge TTS voices to characters based on their profiles"""
        voice_index = {'male': 0, 'female': 0}
        
        for char_id, character in self.characters.items():
            gender = character.get('gender', 'male').lower()
            
            if gender in self.edge_voices and isinstance(self.edge_voices[gender], list):
                # Rotate through available voices for variety
                voice_list = self.edge_voices[gender]
                voice = voice_list[voice_index[gender] % len(voice_list)]
                voice_index[gender] += 1
            else:
                # Default voice
                voice = 'en-US-AriaNeural' if gender == 'female' else 'en-US-GuyNeural'
            
            self.character_voice_mapping[char_id] = voice
            logger.info(f"Assigned voice '{voice}' to character '{character.get('name', char_id)}'")
        
        # Add narrator voice
        self.character_voice_mapping['char_narrator'] = self.edge_voices['narrator']
        logger.info(f"Assigned narrator voice: {self.edge_voices['narrator']}")

    def get_voice_settings(self, character_id: str, emotion: str = "neutral") -> Dict[str, Any]:
        """Get voice settings for a character based on emotion and personality"""
        character = self.characters.get(character_id, {})
        personality = character.get('personality', {})
        
        # Base settings
        rate = "+0%"  # Normal speed
        pitch = "+0Hz"  # Normal pitch
        volume = "+0%"  # Normal volume
        
        # Adjust based on personality traits
        extraversion = personality.get('extraversion', 0.5)
        neuroticism = personality.get('neuroticism', 0.5)
        confidence = personality.get('confidence_score', 0.5)
        
        # Adjust rate based on extraversion (more extroverted = faster speech)
        if extraversion > 0.7:
            rate = "+10%"
        elif extraversion < 0.3:
            rate = "-10%"
        
        # Adjust pitch based on emotion and confidence
        if emotion in ['excited', 'happy', 'amazed']:
            pitch = "+50Hz"
        elif emotion in ['sad', 'vulnerable', 'reflective']:
            pitch = "-30Hz"
        elif emotion in ['angry', 'demanding', 'challenging']:
            pitch = "+20Hz"
        elif confidence > 0.8:
            pitch = "+10Hz"
        elif confidence < 0.3:
            pitch = "-20Hz"
        
        # Adjust volume based on emotion
        if emotion in ['angry', 'demanding', 'challenging', 'brave']:
            volume = "+20%"
        elif emotion in ['whispered', 'vulnerable', 'mysterious']:
            volume = "-10%"
        
        return {
            'rate': rate,
            'pitch': pitch,
            'volume': volume
        }

    async def generate_audio_for_text(self, text: str, voice: str, output_file: Path, 
                                    voice_settings: Dict[str, Any] = None):
        """Generate audio file for given text using Edge TTS"""
        try:
            if voice_settings:
                # Apply voice modifications
                rate = voice_settings.get('rate', '+0%')
                pitch = voice_settings.get('pitch', '+0Hz')
                volume = voice_settings.get('volume', '+0%')
                
                # Create SSML with voice settings
                ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                    <voice name="{voice}">
                        <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
                            {text}
                        </prosody>
                    </voice>
                </speak>"""
                
                communicate = edge_tts.Communicate(ssml, voice, pitch=pitch, rate=rate, volume=volume)
            else:
                communicate = edge_tts.Communicate(text, voice)
            
            await communicate.save(str(output_file))
            logger.info(f"Generated audio: {output_file.name}")
            
        except Exception as e:
            logger.error(f"Error generating audio for '{text[:50]}...': {e}")
            raise

    async def generate_dialogue_audio(self):
        """Generate audio files for all dialogue lines"""
        dialogue_dir = self.audio_dir / "dialogue"
        dialogue_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating audio for {len(self.dialogue)} dialogue lines...")
        
        for dialogue_id, dialogue_data in self.dialogue.items():
            character_id = dialogue_data.get('character_id')
            text = dialogue_data.get('text', '')
            emotion = dialogue_data.get('emotion', 'neutral')
            
            if not character_id or not text:
                continue
            
            # Get assigned voice
            voice = self.character_voice_mapping.get(character_id, 'en-US-AriaNeural')
            
            # Get voice settings based on character and emotion
            voice_settings = self.get_voice_settings(character_id, emotion)
            
            # Create output filename
            safe_dialogue_id = re.sub(r'[^\w\-_\.]', '_', dialogue_id)
            output_file = dialogue_dir / f"{safe_dialogue_id}.wav"
            
            # Generate audio
            await self.generate_audio_for_text(text, voice, output_file, voice_settings)
            
            # Update dialogue data with audio file path
            dialogue_data['audio_file_path'] = str(output_file.relative_to(self.data_dir))

    async def generate_chapter_summaries_audio(self):
        """Generate audio files for chapter summaries"""
        chapters_dir = self.audio_dir / "chapters"
        chapters_dir.mkdir(exist_ok=True)
        
        narrator_voice = self.edge_voices['narrator']
        
        logger.info(f"Generating chapter summary audio for {len(self.chapters)} chapters...")
        
        for chapter_id, chapter_data in self.chapters.items():
            title = chapter_data.get('title', f'Chapter {chapter_data.get("number", "")}')
            summary = chapter_data.get('summary', '')
            
            if not summary:
                continue
            
            # Combine title and summary
            full_text = f"{title}. {summary}"
            
            # Create output filename
            chapter_num = chapter_data.get('number', '0')
            output_file = chapters_dir / f"chapter_{chapter_num:02d}_summary.wav"
            
            # Generate audio with narrator voice
            voice_settings = {'rate': '+0%', 'pitch': '-10Hz', 'volume': '+10%'}  # Authoritative narrator tone
            await self.generate_audio_for_text(full_text, narrator_voice, output_file, voice_settings)

    async def generate_full_audiobook_sample(self):
        """Generate a sample audiobook combining chapter intros and key dialogue"""
        sample_dir = self.audio_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        logger.info("Generating full audiobook sample...")
        
        # Create text combining chapter summaries and dialogue
        full_text_parts = []
        
        # Sort chapters by number
        sorted_chapters = sorted(self.chapters.items(), 
                               key=lambda x: x[1].get('number', 0))
        
        for chapter_id, chapter_data in sorted_chapters[:3]:  # First 3 chapters for sample
            chapter_num = chapter_data.get('number', 0)
            title = chapter_data.get('title', f'Chapter {chapter_num}')
            summary = chapter_data.get('summary', '')
            
            # Add chapter header
            full_text_parts.append(f"\\n\\n{title}\\n\\n{summary}")
            
            # Add some dialogue from this chapter
            chapter_dialogue = [dlg for dlg in self.dialogue.values() 
                              if dlg.get('chapter') == chapter_num]
            
            if chapter_dialogue:
                full_text_parts.append("\\n\\nKey scenes from this chapter:\\n")
                for dlg in chapter_dialogue[:3]:  # First 3 dialogue lines per chapter
                    character_name = self.characters.get(dlg.get('character_id', ''), {}).get('name', 'Unknown')
                    if dlg.get('character_id') == 'char_narrator':
                        full_text_parts.append(f"Narrator: {dlg.get('text', '')}")
                    else:
                        full_text_parts.append(f"{character_name}: {dlg.get('text', '')}")
        
        # Combine all parts
        full_text = " ".join(full_text_parts)
        
        if full_text.strip():
            output_file = sample_dir / "audiobook_sample.wav"
            narrator_voice = self.edge_voices['narrator']
            voice_settings = {'rate': '+5%', 'pitch': '-5Hz', 'volume': '+15%'}
            
            await self.generate_audio_for_text(full_text, narrator_voice, output_file, voice_settings)
            logger.info(f"Generated full audiobook sample: {output_file}")

    async def update_json_with_audio_paths(self):
        """Update JSON files with generated audio file paths"""
        try:
            # Update dialogue.json with audio paths
            dialogue_output_file = self.json_dir / "dialogue.json"
            if dialogue_output_file.exists():
                dialogue_list = list(self.dialogue.values())
                updated_data = {
                    "metadata": {
                        "total_dialogue_lines": len(dialogue_list),
                        "generated_at": str(Path().resolve()),
                        "source": "audio_generation_updated"
                    },
                    "dialogue": dialogue_list
                }
                
                with open(dialogue_output_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2)
                
                logger.info("Updated dialogue.json with audio file paths")
            
        except Exception as e:
            logger.error(f"Error updating JSON files: {e}")

    def create_audio_index(self):
        """Create an index of all generated audio files"""
        try:
            audio_index = {
                "metadata": {
                    "generated_at": str(Path().resolve()),
                    "total_characters": len(self.character_voice_mapping),
                    "total_dialogue_lines": len(self.dialogue),
                    "total_chapters": len(self.chapters)
                },
                "character_voice_mapping": self.character_voice_mapping,
                "audio_files": {
                    "dialogue": [],
                    "chapters": [],
                    "samples": []
                }
            }
            
            # List all generated audio files
            for category in ["dialogue", "chapters", "samples"]:
                category_dir = self.audio_dir / category
                if category_dir.exists():
                    for audio_file in category_dir.glob("*.wav"):
                        audio_index["audio_files"][category].append({
                            "filename": audio_file.name,
                            "path": str(audio_file.relative_to(self.data_dir)),
                            "size_bytes": audio_file.stat().st_size if audio_file.exists() else 0
                        })
            
            # Save index
            index_file = self.audio_dir / "audio_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(audio_index, f, indent=2)
            
            logger.info(f"Created audio index: {index_file}")
            
        except Exception as e:
            logger.error(f"Error creating audio index: {e}")

    async def generate_all_audio(self):
        """Generate all audio files"""
        print("EchoTales Voice Audio Generator")
        print("=" * 50)
        
        try:
            # Load data
            logger.info("Loading JSON data...")
            self.load_json_data()
            
            # Assign voices
            logger.info("Assigning voices to characters...")
            self.assign_voices_to_characters()
            
            # Generate dialogue audio
            logger.info("Generating dialogue audio files...")
            await self.generate_dialogue_audio()
            
            # Generate chapter summaries
            logger.info("Generating chapter summary audio files...")
            await self.generate_chapter_summaries_audio()
            
            # Generate full sample
            logger.info("Generating audiobook sample...")
            await self.generate_full_audiobook_sample()
            
            # Update JSON files
            logger.info("Updating JSON files with audio paths...")
            await self.update_json_with_audio_paths()
            
            # Create audio index
            logger.info("Creating audio file index...")
            self.create_audio_index()
            
            print("\\n" + "=" * 60)
            print("AUDIO GENERATION SUMMARY")
            print("=" * 60)
            print(f"✅ Characters with voices: {len(self.character_voice_mapping)}")
            print(f"✅ Dialogue audio files: {len(self.dialogue)}")
            print(f"✅ Chapter summaries: {len(self.chapters)}")
            print(f"✅ Audio samples: 1")
            print(f"\\nAudio files saved to: {self.audio_dir}")
            print("=" * 60)
            
            logger.info("Audio generation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in audio generation: {e}")
            print(f"\\n❌ Error: {e}")
            sys.exit(1)

def main():
    """Main entry point"""
    generator = VoiceAudioGenerator()
    asyncio.run(generator.generate_all_audio())

if __name__ == "__main__":
    main()