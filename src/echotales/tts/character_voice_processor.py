#!/usr/bin/env python3
"""
Character Voice Processor for Dummy Data
Processes character data and creates consistent voice mappings for TTS generation
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
from collections import defaultdict

from .enhanced_edge_tts import EnhancedEdgeTTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharacterVoiceProcessor:
    """Process character data and create consistent voice mappings"""
    
    def __init__(self, dummy_data_path: str = None):
        self.dummy_data_path = dummy_data_path or self._get_default_dummy_data_path()
        self.enhanced_tts = EnhancedEdgeTTS()
        self.character_data = {}
        self.chapter_data = {}
        self.voice_assignments = {}
        
    def _get_default_dummy_data_path(self) -> str:
        """Get default path to dummy data"""
        current_dir = Path(__file__).parent.parent.parent.parent
        return str(current_dir / "dummy_data")
    
    async def load_dummy_data(self):
        """Load character and chapter data from dummy files"""
        dummy_path = Path(self.dummy_data_path)
        
        # Load character data
        characters_file = dummy_path / "advanced_characters.json"
        if characters_file.exists():
            with open(characters_file, 'r', encoding='utf-8') as f:
                self.character_data = json.load(f)
            logger.info(f"Loaded {len(self.character_data.get('characters', []))} characters")
        
        # Load chapter data
        chapter_files = list(dummy_path.glob("samples_chapter*.json"))
        for chapter_file in chapter_files:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapter_data = json.load(f)
                chapter_num = chapter_data.get('chapter', 1)
                self.chapter_data[chapter_num] = chapter_data
        
        logger.info(f"Loaded {len(self.chapter_data)} chapters")
        
        # Initialize TTS system
        await self.enhanced_tts.initialize()
    
    def analyze_characters(self) -> Dict:
        """Analyze characters and group by gender/age for consistent voice assignment"""
        if not self.character_data:
            return {}
        
        character_groups = defaultdict(list)
        character_info = {}
        
        for character in self.character_data.get('characters', []):
            char_id = character['character_id']
            name = character['name']
            gender = character.get('gender', 'Unknown').lower()
            age_group = character.get('age_group', 'Adult').lower().replace(' ', '_')
            
            # Normalize gender
            if gender == 'unknown':
                # Try to infer from name or default to male
                gender = self._infer_gender_from_name(name)
            
            # Map age groups to our system
            age_mapping = {
                'young_adult': 'young_adult',
                'adult': 'middle_aged', 
                'middle_aged': 'middle_aged',
                'elderly': 'elderly',
                'child': 'child',
                'teenager': 'young_adult'
            }
            age_group = age_mapping.get(age_group, 'young_adult')
            
            # Create character info
            char_info = {
                'character_id': char_id,
                'name': name,
                'gender': gender,
                'age_group': age_group,
                'mention_count': character.get('mention_count', 0),
                'is_proper_name': character.get('is_proper_name', False),
                'audio_file': character.get('audio_file', ''),
                'personality': character.get('personality', {}),
                'character_type': self._determine_character_type(character)
            }
            
            character_info[char_id] = char_info
            
            # Group by gender and age for consistent voice assignment
            group_key = f"{gender}_{age_group}"
            character_groups[group_key].append(char_info)
        
        return {
            'character_info': character_info,
            'character_groups': dict(character_groups)
        }
    
    def _infer_gender_from_name(self, name: str) -> str:
        """Simple gender inference from character name"""
        name_lower = name.lower()
        
        # Common male indicators
        male_indicators = ['mr', 'sir', 'lord', 'king', 'prince', 'duke', 'baron', 'zhou', 'klein']
        # Common female indicators  
        female_indicators = ['ms', 'mrs', 'miss', 'lady', 'queen', 'princess', 'duchess']
        
        for indicator in male_indicators:
            if indicator in name_lower:
                return 'male'
        
        for indicator in female_indicators:
            if indicator in name_lower:
                return 'female'
        
        # Default to male if uncertain
        return 'male'
    
    def _determine_character_type(self, character: Dict) -> str:
        """Determine character type based on character data"""
        name = character.get('name', '').lower()
        mention_count = character.get('mention_count', 0)
        is_proper_name = character.get('is_proper_name', False)
        
        # Main character indicators
        if mention_count > 1000 or name in ['klein', 'i', 'zhou mingrui']:
            return 'protagonist'
        
        # Narrator
        if name == 'narrator':
            return 'narrator'
        
        # Important characters (high mention count)
        if mention_count > 500:
            return 'mentor'
        
        # Regular characters
        if mention_count > 100:
            return 'supporting'
        
        # Minor characters
        return 'minor'
    
    async def assign_voices(self, character_analysis: Dict) -> Dict:
        """Assign consistent voices to characters based on gender/age groups"""
        character_groups = character_analysis['character_groups']
        character_info = character_analysis['character_info']
        voice_assignments = {}
        
        # Get available voices by category
        available_voices = await self.enhanced_tts.list_character_voices()
        
        for group_key, characters in character_groups.items():
            gender, age_group = group_key.split('_', 1)
            
            # Get available voices for this gender
            gender_voices = available_voices.get(gender, [])
            if not gender_voices:
                # Fallback to opposite gender if needed
                fallback_gender = 'female' if gender == 'male' else 'male'
                gender_voices = available_voices.get(fallback_gender, [])
            
            if not gender_voices:
                # Ultimate fallback
                gender_voices = ['en-US-AriaNeural', 'en-US-BrandonNeural']
            
            # Sort characters by importance (mention count)
            characters.sort(key=lambda x: x['mention_count'], reverse=True)
            
            # Assign voices consistently
            for i, character in enumerate(characters):
                char_id = character['character_id']
                
                # Use hash-based selection for consistency
                voice_index = hash(f"{char_id}_{gender}_{age_group}") % len(gender_voices)
                selected_voice = gender_voices[voice_index]
                
                voice_assignments[char_id] = {
                    'character_info': character,
                    'assigned_voice': selected_voice,
                    'voice_settings': {
                        'gender': gender,
                        'age_group': age_group,
                        'character_type': character['character_type'],
                        'personality_traits': self._extract_personality_traits(character['personality'])
                    }
                }
                
                logger.info(f"Assigned voice '{selected_voice}' to character '{character['name']}' "
                          f"({gender}, {age_group}, {character['character_type']})")
        
        return voice_assignments
    
    def _extract_personality_traits(self, personality: Dict) -> List[str]:
        """Extract personality traits from personality scores"""
        traits = []
        
        # Map personality dimensions to traits
        if personality.get('extraversion', 0.5) > 0.6:
            traits.append('friendly')
        if personality.get('openness', 0.5) > 0.6:
            traits.append('creative')
        if personality.get('conscientiousness', 0.5) > 0.6:
            traits.append('serious')
        if personality.get('agreeableness', 0.5) > 0.6:
            traits.append('warm')
        if personality.get('neuroticism', 0.5) > 0.6:
            traits.append('nervous')
        
        return traits if traits else ['friendly']
    
    async def generate_chapter_audio(self, chapter_number: int, output_dir: str = "generated_audio") -> bool:
        """Generate concatenated audio for a specific chapter"""
        if chapter_number not in self.chapter_data:
            logger.error(f"Chapter {chapter_number} not found in data")
            return False
        
        chapter_data = self.chapter_data[chapter_number]
        lines = chapter_data.get('lines', [])
        
        if not lines:
            logger.warning(f"No lines found in chapter {chapter_number}")
            return False
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group lines by character to maintain voice consistency
        character_lines = defaultdict(list)
        for line in lines:
            char_id = line['character_id']
            character_lines[char_id].append(line)
        
        # Generate audio for each character's lines
        audio_files = []
        
        for char_id, char_lines in character_lines.items():
            if char_id not in self.voice_assignments:
                logger.warning(f"No voice assignment found for character {char_id}")
                continue
            
            voice_assignment = self.voice_assignments[char_id]
            character_info = voice_assignment['character_info']
            voice_settings = voice_assignment['voice_settings']
            
            # Concatenate all lines for this character
            character_text_segments = []
            for line in char_lines:
                text = line['text'].strip()
                emotion = line.get('emotion', 'neutral')
                
                # Add emotion context to text if needed
                if emotion != 'neutral':
                    character_text_segments.append(f"[{emotion}] {text}")
                else:
                    character_text_segments.append(text)
            
            if not character_text_segments:
                continue
            
            # Create character-specific audio file
            char_name = character_info['name'].replace(' ', '_').replace('/', '_')
            char_filename = f"ch{chapter_number:04d}_{char_name}_{char_id}.wav"
            char_output_path = output_path / char_filename
            
            # Generate audio for this character's lines
            full_text = ' ... '.join(character_text_segments)
            
            success = await self.enhanced_tts.synthesize_character_speech(
                text=full_text,
                character_name=character_info['name'],
                output_path=str(char_output_path),
                gender=voice_settings['gender'],
                age_group=voice_settings['age_group'],
                character_type=voice_settings['character_type'],
                personality_traits=voice_settings['personality_traits'],
                apply_modifications=True
            )
            
            if success:
                audio_files.append({
                    'character_id': char_id,
                    'character_name': character_info['name'],
                    'file_path': str(char_output_path),
                    'line_count': len(char_lines)
                })
                logger.info(f"Generated audio for {character_info['name']}: {char_filename}")
            else:
                logger.error(f"Failed to generate audio for {character_info['name']}")
        
        # Create chapter summary
        chapter_summary = {
            'chapter': chapter_number,
            'total_characters': len(character_lines),
            'audio_files': audio_files,
            'output_directory': str(output_path)
        }
        
        # Save summary
        summary_file = output_path / f"chapter_{chapter_number:04d}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(chapter_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Chapter {chapter_number} audio generation complete. "
                   f"Generated {len(audio_files)} character audio files.")
        
        return True
    
    async def process_all_chapters(self, output_dir: str = "generated_audio") -> Dict:
        """Process all available chapters and generate audio"""
        results = {}
        
        for chapter_num in self.chapter_data.keys():
            logger.info(f"Processing chapter {chapter_num}...")
            success = await self.generate_chapter_audio(chapter_num, output_dir)
            results[chapter_num] = success
        
        return results
    
    async def run_complete_processing(self) -> Dict:
        """Run the complete processing pipeline"""
        logger.info("Starting complete character voice processing...")
        
        # Load data
        await self.load_dummy_data()
        
        # Analyze characters
        logger.info("Analyzing characters...")
        character_analysis = self.analyze_characters()
        
        # Assign voices
        logger.info("Assigning voices...")
        self.voice_assignments = await self.assign_voices(character_analysis)
        
        # Generate audio for all chapters
        logger.info("Generating chapter audio...")
        processing_results = await self.process_all_chapters()
        
        return {
            'character_analysis': character_analysis,
            'voice_assignments': self.voice_assignments,
            'processing_results': processing_results
        }


async def main():
    """Test the character voice processor"""
    print("EchoTales Character Voice Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = CharacterVoiceProcessor()
    
    # Run complete processing
    results = await processor.run_complete_processing()
    
    # Print results summary
    print(f"\n📊 Processing Results:")
    print(f"Characters analyzed: {len(results['character_analysis']['character_info'])}")
    print(f"Voice assignments: {len(results['voice_assignments'])}")
    print(f"Chapters processed: {len(results['processing_results'])}")
    
    # Show character groups
    print(f"\n👥 Character Groups:")
    for group, characters in results['character_analysis']['character_groups'].items():
        print(f"  {group}: {len(characters)} characters")
    
    # Show voice assignments
    print(f"\n🗣️ Voice Assignments (Top 10):")
    for i, (char_id, assignment) in enumerate(list(results['voice_assignments'].items())[:10]):
        char_info = assignment['character_info']
        voice = assignment['assigned_voice']
        print(f"  {char_info['name']} ({char_info['gender']}, {char_info['age_group']}): {voice}")
    
    print(f"\n✅ Character voice processing complete!")
    print(f"🎬 Check 'generated_audio' folder for audio files")


if __name__ == "__main__":
    asyncio.run(main())