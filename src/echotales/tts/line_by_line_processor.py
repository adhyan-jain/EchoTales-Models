#!/usr/bin/env python3
"""
Line-by-Line Chapter Processor for EchoTales
Processes chapter data line by line in correct order and generates sequential audio
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .enhanced_edge_tts import EnhancedEdgeTTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LineByLineProcessor:
    """Process chapter lines in exact order and generate sequential audio"""
    
    def __init__(self, dummy_data_path: str = None):
        self.dummy_data_path = dummy_data_path or self._get_default_dummy_data_path()
        self.enhanced_tts = EnhancedEdgeTTS()
        self.character_data = {}
        self.voice_assignments = {}
        
    def _get_default_dummy_data_path(self) -> str:
        """Get default path to dummy data"""
        current_dir = Path(__file__).parent.parent.parent.parent
        return str(current_dir / "dummy_data")
    
    async def initialize(self):
        """Initialize the processor"""
        await self.enhanced_tts.initialize()
        await self.load_character_data()
        self.create_voice_assignments()
    
    async def load_character_data(self):
        """Load character data from dummy files"""
        dummy_path = Path(self.dummy_data_path)
        
        # Load character data
        characters_file = dummy_path / "advanced_characters.json"
        if characters_file.exists():
            with open(characters_file, 'r', encoding='utf-8') as f:
                self.character_data = json.load(f)
            logger.info(f"Loaded {len(self.character_data.get('characters', []))} characters")
        else:
            logger.warning("Character data not found")
    
    def create_voice_assignments(self):
        """Create voice assignments for characters with smoother voices"""
        # Smoother voice preferences (like Zhou Mingrui's voice)
        smooth_male_voices = [
            "en-US-ChristopherNeural",  # Very smooth and natural
            "en-US-RogerNeural",        # Warm and smooth
            "en-US-SteffanNeural",      # Clear and natural
            "en-US-BrandonNeural"       # Friendly and smooth
        ]
        
        smooth_female_voices = [
            "en-US-JennyNeural",        # Very natural and smooth
            "en-US-MichelleNeural",     # Warm and clear
            "en-US-AriaNeural",         # Clear and professional
            "en-US-MonicaNeural"        # Gentle and smooth
        ]
        
        narrator_voice = "en-US-ChristopherNeural"  # Smooth narrator voice
        
        # Create character mapping
        character_map = {}
        
        if self.character_data and 'characters' in self.character_data:
            male_voice_index = 0
            female_voice_index = 0
            
            # Sort characters by importance (mention count)
            characters = sorted(
                self.character_data['characters'], 
                key=lambda x: x.get('mention_count', 0), 
                reverse=True
            )
            
            for character in characters:
                char_id = character['character_id']
                name = character['name']
                gender = character.get('gender', 'Unknown').lower()
                age_group = character.get('age_group', 'Adult').lower().replace(' ', '_')
                mention_count = character.get('mention_count', 0)
                
                # Normalize gender
                if gender == 'unknown':
                    gender = self._infer_gender_from_name(name)
                
                # Map age groups
                age_mapping = {
                    'young_adult': 'young_adult',
                    'adult': 'middle_aged', 
                    'middle_aged': 'middle_aged',
                    'elderly': 'elderly',
                    'child': 'child',
                    'teenager': 'young_adult'
                }
                age_group = age_mapping.get(age_group, 'middle_aged')
                
                # Determine character type
                character_type = self._determine_character_type(character)
                
                # Assign voice based on gender
                if gender == 'male':
                    voice = smooth_male_voices[male_voice_index % len(smooth_male_voices)]
                    male_voice_index += 1
                else:
                    voice = smooth_female_voices[female_voice_index % len(smooth_female_voices)]
                    female_voice_index += 1
                
                character_map[char_id] = {
                    'name': name,
                    'voice': voice,
                    'gender': gender,
                    'age_group': age_group,
                    'character_type': character_type,
                    'personality_traits': self._extract_personality_traits(character.get('personality', {}))
                }
                
                logger.info(f"Assigned {voice} to {name} ({gender}, {age_group}, {character_type})")
        
        # Add narrator mapping
        character_map['char_narrator'] = {
            'name': 'Narrator',
            'voice': narrator_voice,
            'gender': 'male',
            'age_group': 'middle_aged',
            'character_type': 'narrator',
            'personality_traits': ['warm', 'authoritative']
        }
        
        self.voice_assignments = character_map
        logger.info(f"Created voice assignments for {len(character_map)} characters")
    
    def _infer_gender_from_name(self, name: str) -> str:
        """Simple gender inference from character name"""
        name_lower = name.lower()
        
        male_indicators = ['mr', 'sir', 'lord', 'king', 'prince', 'duke', 'baron', 'zhou', 'klein']
        female_indicators = ['ms', 'mrs', 'miss', 'lady', 'queen', 'princess', 'duchess']
        
        for indicator in male_indicators:
            if indicator in name_lower:
                return 'male'
        
        for indicator in female_indicators:
            if indicator in name_lower:
                return 'female'
        
        return 'male'  # Default
    
    def _determine_character_type(self, character: Dict) -> str:
        """Determine character type"""
        name = character.get('name', '').lower()
        mention_count = character.get('mention_count', 0)
        
        if mention_count > 1000 or name in ['klein', 'i', 'zhou mingrui']:
            return 'protagonist'
        elif name == 'narrator':
            return 'narrator'
        elif mention_count > 500:
            return 'mentor'
        elif mention_count > 100:
            return 'supporting'
        else:
            return 'minor'
    
    def _extract_personality_traits(self, personality: Dict) -> List[str]:
        """Extract personality traits"""
        traits = []
        
        if personality.get('extraversion', 0.5) > 0.6:
            traits.append('friendly')
        if personality.get('openness', 0.5) > 0.6:
            traits.append('warm')
        if personality.get('conscientiousness', 0.5) > 0.6:
            traits.append('serious')
        if personality.get('agreeableness', 0.5) > 0.6:
            traits.append('warm')
        
        return traits if traits else ['friendly']
    
    async def process_chapter_line_by_line(self, chapter_number: int, output_dir: str = "generated_audio") -> bool:
        """Process chapter line by line in correct order"""
        dummy_path = Path(self.dummy_data_path)
        chapter_file = dummy_path / f"samples_chapter{chapter_number:04d}.json"
        
        if not chapter_file.exists():
            logger.error(f"Chapter file not found: {chapter_file}")
            return False
        
        # Load chapter data
        with open(chapter_file, 'r', encoding='utf-8') as f:
            chapter_data = json.load(f)
        
        lines = chapter_data.get('lines', [])
        if not lines:
            logger.warning(f"No lines found in chapter {chapter_number}")
            return False
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {len(lines)} lines from chapter {chapter_number}")
        
        # Process each line in order
        generated_files = []
        
        for i, line in enumerate(lines):
            line_id = line['line_id']
            character_id = line['character_id']
            character_name = line['character_name']
            text = line['text'].strip()
            emotion = line.get('emotion', 'neutral')
            
            if not text:
                continue
            
            # Get voice assignment
            if character_id not in self.voice_assignments:
                logger.warning(f"No voice assignment for character {character_id}, using default")
                continue
            
            voice_info = self.voice_assignments[character_id]
            
            # Use raw text without SSML tags to avoid spoken markup
            enhanced_text = text
            
            # Generate filename
            line_filename = f"ch{chapter_number:04d}_line{i+1:03d}_{line_id}_{character_id}.wav"
            line_output_path = output_path / line_filename
            
            # Generate audio for this line
            success = await self.enhanced_tts.synthesize_character_speech(
                text=enhanced_text,
                character_name=voice_info['name'],
                output_path=str(line_output_path),
                gender=voice_info['gender'],
                age_group=voice_info['age_group'],
                character_type=voice_info['character_type'],
                personality_traits=voice_info['personality_traits'],
                apply_modifications=True
            )
            
            if success:
                generated_files.append({
                    'line_number': i + 1,
                    'line_id': line_id,
                    'character_id': character_id,
                    'character_name': voice_info['name'],
                    'voice': voice_info['voice'],
                    'text': text,
                    'emotion': emotion,
                    'file_path': str(line_output_path),
                    'file_size': line_output_path.stat().st_size if line_output_path.exists() else 0
                })
                logger.info(f"Generated line {i+1:03d}: {voice_info['name']} - \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            else:
                logger.error(f"Failed to generate audio for line {i+1}: {line_id}")
        
        # Create chapter summary
        chapter_summary = {
            'chapter': chapter_number,
            'total_lines': len(lines),
            'generated_lines': len(generated_files),
            'processing_success_rate': len(generated_files) / len(lines) if lines else 0,
            'voice_assignments': {char_id: info['voice'] for char_id, info in self.voice_assignments.items()},
            'generated_files': generated_files,
            'output_directory': str(output_path)
        }
        
        # Save summary
        summary_file = output_path / f"chapter_{chapter_number:04d}_line_by_line_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(chapter_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Chapter {chapter_number} line-by-line processing complete.")
        logger.info(f"Generated {len(generated_files)}/{len(lines)} audio files")
        logger.info(f"Success rate: {chapter_summary['processing_success_rate']*100:.1f}%")
        
        return len(generated_files) > 0
    
    async def concatenate_chapter_lines(self, chapter_number: int, 
                                      output_dir: str = "generated_audio",
                                      silence_duration: float = 0.5) -> Optional[str]:
        """Concatenate all line audio files into a single chapter file"""
        output_path = Path(output_dir)
        summary_file = output_path / f"chapter_{chapter_number:04d}_line_by_line_summary.json"
        
        if not summary_file.exists():
            logger.error(f"Chapter summary not found: {summary_file}")
            return None
        
        # Load summary
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        generated_files = summary.get('generated_files', [])
        if not generated_files:
            logger.warning(f"No generated files found for chapter {chapter_number}")
            return None
        
        try:
            from pydub import AudioSegment
            
            # Create concatenated audio
            concatenated = AudioSegment.empty()
            silence = AudioSegment.silent(duration=int(silence_duration * 1000))
            
            for file_info in generated_files:
                file_path = Path(file_info['file_path'])
                if file_path.exists():
                    try:
                        audio = AudioSegment.from_wav(str(file_path))
                        
                        # Add silence before (except for first line)
                        if len(concatenated) > 0:
                            concatenated += silence
                        
                        concatenated += audio
                        logger.debug(f"Added line {file_info['line_number']}: {file_info['character_name']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load audio file {file_path}: {e}")
                else:
                    logger.warning(f"Audio file not found: {file_path}")
            
            if len(concatenated) == 0:
                logger.error("No audio was concatenated")
                return None
            
            # Export concatenated audio
            output_filename = f"chapter_{chapter_number:04d}_complete_line_by_line.wav"
            final_output_path = output_path / output_filename
            
            concatenated.export(str(final_output_path), format="wav")
            
            logger.info(f"Created complete chapter audio: {final_output_path}")
            logger.info(f"Total duration: {len(concatenated)/1000:.2f} seconds")
            logger.info(f"Total lines: {len(generated_files)}")
            
            return str(final_output_path)
            
        except ImportError:
            logger.error("PyDub not available - cannot concatenate audio files")
            logger.info("Install with: pip install pydub")
            return None
        except Exception as e:
            logger.error(f"Failed to concatenate chapter audio: {e}")
            return None
    
    async def process_complete_chapter(self, chapter_number: int, output_dir: str = "generated_audio") -> Dict:
        """Process complete chapter with line-by-line generation and concatenation"""
        logger.info(f"Processing complete chapter {chapter_number}...")
        
        # Step 1: Generate line-by-line audio
        line_success = await self.process_chapter_line_by_line(chapter_number, output_dir)
        
        # Step 2: Concatenate lines
        concat_file = None
        if line_success:
            concat_file = await self.concatenate_chapter_lines(chapter_number, output_dir)
        
        return {
            'chapter': chapter_number,
            'line_by_line_success': line_success,
            'concatenated_file': concat_file,
            'success': line_success and concat_file is not None
        }


async def main():
    """Test the line-by-line processor"""
    print("📝 EchoTales Line-by-Line Chapter Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = LineByLineProcessor()
    await processor.initialize()
    
    # Process chapter 1
    print(f"\n🎬 Processing Chapter 1 line by line...")
    result = await processor.process_complete_chapter(1)
    
    # Show results
    print(f"\n📊 PROCESSING RESULTS")
    print("-" * 30)
    print(f"✅ Line-by-line generation: {'Success' if result['line_by_line_success'] else 'Failed'}")
    print(f"✅ Chapter concatenation: {'Success' if result['concatenated_file'] else 'Failed'}")
    
    if result['concatenated_file']:
        file_path = Path(result['concatenated_file'])
        file_size = file_path.stat().st_size / 1024 / 1024  # MB
        print(f"📁 Complete chapter file: {file_path.name} ({file_size:.1f} MB)")
    
    print(f"\n🎉 Line-by-line processing complete!")
    print(f"📁 Check the 'generated_audio' folder for individual line files and complete chapter")


if __name__ == "__main__":
    asyncio.run(main())