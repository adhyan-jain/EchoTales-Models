#!/usr/bin/env python3
"""
Voice Character Mapper and Edge TTS Audio Generator
Maps characters to voice profiles and generates audio using Microsoft Edge TTS
"""

import json
import asyncio
import edge_tts
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import random
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceAudioGenerator:
    """Handles voice mapping and audio generation using Edge TTS"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.json_dir = self.data_dir / "output" / "json"
        self.audio_dir = self.data_dir / "output" / "audio"
        
        # Create audio output directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        (self.audio_dir / "characters").mkdir(exist_ok=True)
        (self.audio_dir / "chapters").mkdir(exist_ok=True)
        (self.audio_dir / "full_books").mkdir(exist_ok=True)
        
        # Edge TTS voice mapping - high quality voices
        self.edge_voices = {
            'male': [
                'en-US-ChristopherNeural',
                'en-US-EricNeural', 
                'en-US-GuyNeural',
                'en-US-RogerNeural',
                'en-US-SteffanNeural',
                'en-GB-RyanNeural',
                'en-GB-ThomasNeural',
                'en-AU-WilliamNeural',
                'en-CA-LiamNeural',
                'en-IN-PrabhatNeural'
            ],
            'female': [
                'en-US-AriaNeural',
                'en-US-AvaNeural',
                'en-US-EmmaNeural',
                'en-US-JennyNeural',
                'en-US-MichelleNeural',
                'en-US-MonicaNeural',
                'en-US-SaraNeural',
                'en-GB-LibbyNeural',
                'en-GB-SoniaNeural',
                'en-AU-NatashaNeural',
                'en-CA-ClaraNeural',
                'en-IN-NeerjaNeural'
            ],
            'narrator': [
                'en-US-AriaNeural',  # Clear, professional
                'en-US-GuyNeural',   # Warm, storytelling voice
                'en-GB-RyanNeural',  # Clear British accent
                'en-US-JennyNeural'  # Versatile, clear
            ]
        }
        
        # Voice characteristics for better mapping
        self.voice_emotions = {
            'cheerful': ['en-US-AriaNeural', 'en-US-EmmaNeural', 'en-US-EricNeural'],
            'calm': ['en-US-SaraNeural', 'en-US-GuyNeural', 'en-GB-RyanNeural'],
            'professional': ['en-US-MonicaNeural', 'en-US-ChristopherNeural'],
            'warm': ['en-US-JennyNeural', 'en-US-SteffanNeural'],
            'authoritative': ['en-US-RogerNeural', 'en-GB-ThomasNeural']
        }
    
    def load_json_files(self) -> Dict[str, Any]:
        """Load all necessary JSON files"""
        logger.info("Loading JSON files...")
        
        files_to_load = {
            'characters': 'characters.json',
            'chapters': 'chapters.json', 
            'dialogue': 'dialogue.json',
            'scenes': 'scenes.json',
            'voice_config': 'voice_config.json',
            'book': 'book.json'
        }
        
        data = {}
        for key, filename in files_to_load.items():
            filepath = self.json_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data[key] = json.load(f)
                    logger.info(f"✅ Loaded {filename}")
                except Exception as e:
                    logger.error(f"❌ Error loading {filename}: {e}")
                    data[key] = None
            else:
                logger.warning(f"⚠️ File not found: {filename}")
                data[key] = None
        
        return data
    
    def create_character_voice_mapping(self, characters_data: Dict, voice_config_data: Dict) -> Dict[str, Dict]:
        """Create mapping between characters and Edge TTS voices"""
        logger.info("Creating character-voice mapping...")
        
        if not characters_data or 'characters' not in characters_data:
            logger.warning("No characters data found, creating default mapping")
            return self._create_default_character_mapping()
        
        character_voice_mapping = {}
        characters = characters_data['characters']
        
        # Track used voices to avoid repetition
        used_male_voices = set()
        used_female_voices = set()
        
        for char in characters:
            char_id = char['character_id']
            char_name = char['name']
            char_gender = char.get('gender', 'unknown').lower()
            
            # Select appropriate voice based on gender and characteristics
            if char_gender == 'male':
                available_voices = [v for v in self.edge_voices['male'] if v not in used_male_voices]
                if not available_voices:
                    available_voices = self.edge_voices['male']  # Reset if all used
                    used_male_voices.clear()
                
                selected_voice = random.choice(available_voices)
                used_male_voices.add(selected_voice)
                
            elif char_gender == 'female':
                available_voices = [v for v in self.edge_voices['female'] if v not in used_female_voices]
                if not available_voices:
                    available_voices = self.edge_voices['female']  # Reset if all used
                    used_female_voices.clear()
                    
                selected_voice = random.choice(available_voices)
                used_female_voices.add(selected_voice)
                
            else:
                # For unknown gender, randomly choose
                selected_voice = random.choice(self.edge_voices['male'] + self.edge_voices['female'])
            
            # Get speaking characteristics
            personality = char.get('personality', {})
            speaking_rate = "medium"
            pitch_adjustment = "+0%"
            
            # Adjust based on personality traits
            if personality.get('extraversion', 0.5) > 0.7:
                speaking_rate = "fast"
                pitch_adjustment = "+10%"
            elif personality.get('extraversion', 0.5) < 0.3:
                speaking_rate = "slow"
                pitch_adjustment = "-10%"
            
            character_voice_mapping[char_id] = {
                'character_name': char_name,
                'edge_voice': selected_voice,
                'gender': char_gender,
                'speaking_rate': speaking_rate,
                'pitch_adjustment': pitch_adjustment,
                'personality_traits': personality.get('primary_traits', []),
                'importance_score': char.get('importance_score', 0.5)
            }
            
            logger.info(f"✅ Mapped {char_name} ({char_gender}) → {selected_voice}")
        
        # Add narrator voice
        narrator_voice = random.choice(self.edge_voices['narrator'])
        character_voice_mapping['char_narrator'] = {
            'character_name': 'Narrator',
            'edge_voice': narrator_voice,
            'gender': 'neutral',
            'speaking_rate': 'medium',
            'pitch_adjustment': '+0%',
            'personality_traits': ['authoritative', 'clear'],
            'importance_score': 1.0
        }
        
        logger.info(f"✅ Added Narrator → {narrator_voice}")
        
        # Save mapping to file
        mapping_file = self.json_dir / "character_voice_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(character_voice_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved character voice mapping to {mapping_file}")
        return character_voice_mapping
    
    def _create_default_character_mapping(self) -> Dict[str, Dict]:
        """Create default character mapping when no character data exists"""
        default_characters = [
            {'id': 'char_001', 'name': 'Alexander', 'gender': 'male'},
            {'id': 'char_002', 'name': 'Isabella', 'gender': 'female'},
            {'id': 'char_003', 'name': 'Marcus', 'gender': 'male'},
            {'id': 'char_004', 'name': 'Elena', 'gender': 'female'},
        ]
        
        mapping = {}
        for i, char in enumerate(default_characters):
            if char['gender'] == 'male':
                voice = self.edge_voices['male'][i % len(self.edge_voices['male'])]
            else:
                voice = self.edge_voices['female'][i % len(self.edge_voices['female'])]
                
            mapping[char['id']] = {
                'character_name': char['name'],
                'edge_voice': voice,
                'gender': char['gender'],
                'speaking_rate': 'medium',
                'pitch_adjustment': '+0%',
                'personality_traits': [],
                'importance_score': 0.8
            }
        
        # Add narrator
        mapping['char_narrator'] = {
            'character_name': 'Narrator',
            'edge_voice': self.edge_voices['narrator'][0],
            'gender': 'neutral',
            'speaking_rate': 'medium',
            'pitch_adjustment': '+0%',
            'personality_traits': ['authoritative'],
            'importance_score': 1.0
        }
        
        return mapping
    
    async def generate_audio_for_text(self, text: str, voice_name: str, 
                                     output_path: Path, speaking_rate: str = "medium",
                                     pitch_adjustment: str = "+0%") -> bool:
        """Generate audio for a single text using Edge TTS"""
        try:
            # Create SSML for better control
            ssml_text = f'''
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{voice_name}">
                    <prosody rate="{speaking_rate}" pitch="{pitch_adjustment}">
                        {text}
                    </prosody>
                </voice>
            </speak>
            '''
            
            # Generate speech
            communicate = edge_tts.Communicate(ssml_text, voice_name)
            await communicate.save(str(output_path))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating audio: {e}")
            return False
    
    async def generate_character_dialogue(self, data: Dict[str, Any], 
                                        character_voice_mapping: Dict[str, Dict]):
        """Generate audio for all character dialogue"""
        logger.info("🎵 Starting dialogue audio generation...")
        
        # Create sample dialogue if none exists
        if not data.get('dialogue') or not data['dialogue'].get('dialogue'):
            logger.warning("No dialogue data found, creating sample dialogue")
            sample_dialogue = self._create_sample_dialogue()
        else:
            sample_dialogue = data['dialogue']['dialogue']
        
        # If still no dialogue, create from characters
        if not sample_dialogue:
            sample_dialogue = self._create_dialogue_from_characters(data.get('characters', {}))
        
        total_lines = len(sample_dialogue)
        logger.info(f"📝 Processing {total_lines} dialogue lines...")
        
        # Generate audio for each dialogue line
        for i, dialogue_line in enumerate(sample_dialogue, 1):
            char_id = dialogue_line.get('character_id', 'char_narrator')
            text = dialogue_line.get('text', '').strip()
            line_id = dialogue_line.get('dialogue_id', f'line_{i:04d}')
            
            if not text:
                continue
            
            # Get voice mapping
            voice_info = character_voice_mapping.get(char_id, character_voice_mapping.get('char_narrator'))
            voice_name = voice_info['edge_voice']
            speaking_rate = voice_info['speaking_rate']
            pitch_adjustment = voice_info['pitch_adjustment']
            char_name = voice_info['character_name']
            
            # Create output filename
            safe_char_name = "".join(c for c in char_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_filename = f"{line_id}_{safe_char_name}.wav"
            output_path = self.audio_dir / "characters" / output_filename
            
            logger.info(f"🎤 [{i}/{total_lines}] Generating: {char_name} - '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            success = await self.generate_audio_for_text(
                text, voice_name, output_path, speaking_rate, pitch_adjustment
            )
            
            if success:
                logger.info(f"✅ Generated: {output_filename}")
            else:
                logger.error(f"❌ Failed: {output_filename}")
    
    def _create_sample_dialogue(self) -> List[Dict]:
        """Create sample dialogue for demonstration"""
        sample_lines = [
            {
                "dialogue_id": "line_0001",
                "character_id": "char_narrator",
                "text": "Welcome to this enchanting tale of adventure and discovery.",
                "emotion": "neutral"
            },
            {
                "dialogue_id": "line_0002", 
                "character_id": "char_001",
                "text": "I never expected to find myself in such an extraordinary place.",
                "emotion": "surprised"
            },
            {
                "dialogue_id": "line_0003",
                "character_id": "char_002",
                "text": "The journey ahead will test our courage and friendship.",
                "emotion": "determined"
            },
            {
                "dialogue_id": "line_0004",
                "character_id": "char_narrator",
                "text": "And so begins our story, where heroes are made and legends are born.",
                "emotion": "dramatic"
            },
            {
                "dialogue_id": "line_0005",
                "character_id": "char_001",
                "text": "Together, we shall overcome any challenge that awaits us.",
                "emotion": "confident"
            }
        ]
        
        return sample_lines
    
    def _create_dialogue_from_characters(self, characters_data: Dict) -> List[Dict]:
        """Create dialogue lines from character data"""
        dialogue_lines = []
        
        if not characters_data or 'characters' not in characters_data:
            return self._create_sample_dialogue()
        
        characters = characters_data['characters']
        
        # Create introduction dialogue for each character
        for i, char in enumerate(characters[:4]):  # Limit to first 4 characters
            char_id = char['character_id']
            char_name = char['name']
            
            dialogue_line = {
                "dialogue_id": f"line_{i+1:04d}",
                "character_id": char_id,
                "text": f"Greetings, I am {char_name}. I'm honored to be part of this tale.",
                "emotion": "neutral"
            }
            dialogue_lines.append(dialogue_line)
        
        # Add narrator introduction
        dialogue_lines.insert(0, {
            "dialogue_id": "line_0000",
            "character_id": "char_narrator", 
            "text": "Welcome to our story. Let me introduce you to our characters.",
            "emotion": "neutral"
        })
        
        return dialogue_lines
    
    async def generate_chapter_audio(self, data: Dict[str, Any], 
                                   character_voice_mapping: Dict[str, Dict]):
        """Generate audio organized by chapters"""
        logger.info("📚 Generating chapter-based audio...")
        
        chapters = data.get('chapters', {}).get('chapters', [])
        
        if not chapters:
            logger.warning("No chapters found, creating sample chapter")
            chapters = [
                {
                    "chapter_id": "chapter_001",
                    "number": 1,
                    "title": "The Beginning",
                    "summary": "Our adventure begins..."
                }
            ]
        
        for chapter in chapters[:3]:  # Process first 3 chapters
            chapter_num = chapter.get('number', 1)
            chapter_title = chapter.get('title', f'Chapter {chapter_num}')
            
            logger.info(f"📖 Processing Chapter {chapter_num}: {chapter_title}")
            
            # Create chapter directory
            chapter_dir = self.audio_dir / "chapters" / f"chapter_{chapter_num:03d}"
            chapter_dir.mkdir(exist_ok=True)
            
            # Generate chapter introduction
            intro_text = f"Chapter {chapter_num}: {chapter_title}. {chapter.get('summary', '')}"
            narrator_voice = character_voice_mapping['char_narrator']['edge_voice']
            
            intro_path = chapter_dir / "00_chapter_intro.wav"
            await self.generate_audio_for_text(intro_text, narrator_voice, intro_path)
            logger.info(f"✅ Generated chapter intro: {intro_path.name}")
    
    async def create_full_audiobook_sample(self, character_voice_mapping: Dict[str, Dict]):
        """Create a sample audiobook combining all elements"""
        logger.info("🎧 Creating full audiobook sample...")
        
        sample_script = [
            ("char_narrator", "Welcome to EchoTales Enhanced. This is a demonstration of our advanced text-to-speech system."),
            ("char_narrator", "Our system can generate natural-sounding speech for multiple characters with distinct voices."),
            ("char_001", "Hello! I'm the first character. Notice how my voice is different from the narrator."),
            ("char_002", "And I'm the second character with my own unique voice and speaking style."),
            ("char_narrator", "Each character can express different emotions and speaking patterns."),
            ("char_001", "I'm excited to be part of this amazing story!"),
            ("char_002", "Together, we create an immersive listening experience for audiobook lovers."),
            ("char_narrator", "This concludes our demonstration. Thank you for listening to EchoTales Enhanced.")
        ]
        
        full_audio_dir = self.audio_dir / "full_books"
        full_audio_dir.mkdir(exist_ok=True)
        
        for i, (char_id, text) in enumerate(sample_script, 1):
            voice_info = character_voice_mapping.get(char_id, character_voice_mapping['char_narrator'])
            voice_name = voice_info['edge_voice']
            char_name = voice_info['character_name']
            
            output_path = full_audio_dir / f"{i:02d}_{char_name.replace(' ', '_')}.wav"
            
            logger.info(f"🎵 [{i}/{len(sample_script)}] Recording: {char_name} - '{text[:40]}...'")
            
            success = await self.generate_audio_for_text(text, voice_name, output_path)
            if success:
                logger.info(f"✅ Generated: {output_path.name}")
    
    def generate_audio_report(self, character_voice_mapping: Dict[str, Dict]):
        """Generate a report of the audio generation process"""
        logger.info("📋 Generating audio report...")
        
        report = {
            "generation_timestamp": "2025-09-22T10:38:00Z",
            "total_characters": len(character_voice_mapping),
            "voice_distribution": {},
            "character_details": character_voice_mapping,
            "audio_files_generated": {
                "characters": len(list((self.audio_dir / "characters").glob("*.wav"))),
                "chapters": len(list((self.audio_dir / "chapters").glob("**/*.wav"))),
                "full_books": len(list((self.audio_dir / "full_books").glob("*.wav")))
            }
        }
        
        # Count voice distribution
        for char_info in character_voice_mapping.values():
            voice = char_info['edge_voice']
            gender = char_info['gender']
            report["voice_distribution"][voice] = report["voice_distribution"].get(voice, 0) + 1
        
        # Save report
        report_file = self.audio_dir / "audio_generation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Audio report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎵 AUDIO GENERATION SUMMARY")
        print("="*60)
        print(f"📊 Characters processed: {len(character_voice_mapping)}")
        print(f"🎤 Unique voices used: {len(report['voice_distribution'])}")
        print(f"🗣️ Audio files generated:")
        for category, count in report["audio_files_generated"].items():
            print(f"   • {category}: {count} files")
        print(f"📂 Audio output directory: {self.audio_dir}")
        print("="*60)


async def main():
    """Main function to run the voice audio generator"""
    print("🎵 EchoTales Voice Audio Generator")
    print("=" * 50)
    
    generator = VoiceAudioGenerator()
    
    try:
        # Load JSON data
        data = generator.load_json_files()
        
        # Create character-voice mapping
        character_voice_mapping = generator.create_character_voice_mapping(
            data['characters'], data['voice_config']
        )
        
        # Generate audio files
        await generator.generate_character_dialogue(data, character_voice_mapping)
        await generator.generate_chapter_audio(data, character_voice_mapping)
        await generator.create_full_audiobook_sample(character_voice_mapping)
        
        # Generate report
        generator.generate_audio_report(character_voice_mapping)
        
        print("\n✅ Audio generation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during audio generation: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())