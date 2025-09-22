#!/usr/bin/env python3
"""
Fixed Dialogue Processor with Chapter Detection
Extracts ALL lines with proper character attribution and chapter separation
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedDialogueProcessor:
    def __init__(self, output_dir: str = "modelsbooknlp/output", dialogue_dir: str = "data/processed/dialogues"):
        self.output_dir = Path(output_dir)
        self.dialogue_dir = Path(dialogue_dir)
        self.dialogue_dir.mkdir(exist_ok=True)
        
        # Emotion detection patterns
        self.emotion_patterns = {
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'damn', 'hell', 'cursed', 'swore', 'shouted'],
            'fear': ['afraid', 'scared', 'terrified', 'fear', 'panic', 'horror', 'dread', 'anxiety', 'nervous'],
            'joy': ['happy', 'glad', 'joy', 'cheerful', 'delighted', 'pleased', 'excited', 'thrilled', 'elated'],
            'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'sorrow', 'grief', 'tears', 'crying', 'mourning'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered', 'startled'],
            'disgust': ['disgusted', 'revolted', 'sickened', 'nauseated', 'repulsed', 'horrified', 'appalled'],
            'curious': ['curious', 'wondering', 'questioning', 'inquisitive', 'intrigued', 'interested', 'investigating'],
            'pain': ['painful', 'hurt', 'pain', 'ache', 'throbbing', 'excruciating', 'agony', 'suffering'],
            'confusion': ['confused', 'bewildered', 'puzzled', 'perplexed', 'lost', 'unclear', 'uncertain'],
            'neutral': ['said', 'spoke', 'told', 'asked', 'replied', 'answered', 'stated', 'mentioned']
        }

    def load_characters(self, characters_file: str = "data/processed/characters/advanced_characters.json") -> Dict[str, Dict]:
        """Load corrected character information"""
        char_path = Path(characters_file)
        if not char_path.exists():
            # Fallback to other locations if not found in data folder
            fallback_paths = [
                Path("characters/advanced_characters.json"),
                Path("characters/enhanced_characters_gemini.json"),
                Path("dummy_data/advanced_characters.json")
            ]
            for fallback in fallback_paths:
                if fallback.exists():
                    char_path = fallback
                    break
            
        if not char_path.exists():
            raise FileNotFoundError(f"Characters file not found in any location: {characters_file}")
            
        with open(char_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create lookup by character_id and booknlp_id
        characters = {}
        for char in data['characters']:
            characters[char['character_id']] = char
            characters[char['booknlp_id']] = char
            
        logger.info(f"Loaded {len(data['characters'])} characters")
        return characters

    def load_booknlp_data(self, book_id: str) -> Dict[str, pd.DataFrame]:
        """Load BookNLP output files"""
        files = {
            'entities': f"{book_id}.entities",
            'tokens': f"{book_id}.tokens"
        }
        
        data = {}
        for key, filename in files.items():
            filepath = self.output_dir / filename
            if filepath.exists():
                if key == 'tokens':
                    data[key] = pd.read_csv(filepath, sep='\t', quoting=3, on_bad_lines='skip')
                else:
                    data[key] = pd.read_csv(filepath, sep='\t', on_bad_lines='skip', quoting=3)
                logger.info(f"Loaded {key}: {len(data[key])} entries")
            else:
                logger.warning(f"File not found: {filepath}")
                
        return data

    def detect_chapter_boundaries(self, tokens_df: pd.DataFrame) -> Dict[int, int]:
        """Detect chapter start markers and return chapter boundaries"""
        logger.info("Detecting chapter boundaries...")
        
        chapter_boundaries = {}
        
        # Find tokens that match chapter start pattern
        chapter_tokens = tokens_df[
            tokens_df['word'].str.contains(r'CHAPTER_START_\d+', na=False, regex=True)
        ]
        
        logger.info(f"Found {len(chapter_tokens)} chapter markers")
        
        for _, token in chapter_tokens.iterrows():
            # Extract chapter number from token
            match = re.search(r'CHAPTER_START_(\d+)', str(token['word']))
            if match:
                chapter_num = int(match.group(1))
                token_id = int(token['token_ID_within_document'])
                chapter_boundaries[chapter_num] = token_id
                logger.debug(f"Chapter {chapter_num} starts at token {token_id}")
                
        # Sort by chapter number
        chapter_boundaries = dict(sorted(chapter_boundaries.items()))
        
        logger.info(f"Detected {len(chapter_boundaries)} chapters: {list(chapter_boundaries.keys())}")
        return chapter_boundaries

    def assign_tokens_to_chapters(self, tokens_df: pd.DataFrame, chapter_boundaries: Dict[int, int]) -> Dict[int, pd.DataFrame]:
        """Assign tokens to their respective chapters"""
        logger.info("Assigning tokens to chapters...")
        
        if not chapter_boundaries:
            # If no chapters found, put everything in chapter 1
            logger.warning("No chapter boundaries found, assigning all content to chapter 1")
            return {1: tokens_df}
        
        chapters = {}
        chapter_numbers = sorted(chapter_boundaries.keys())
        
        for i, chapter_num in enumerate(chapter_numbers):
            start_token = chapter_boundaries[chapter_num]
            
            # Determine end token (start of next chapter or end of document)
            if i + 1 < len(chapter_numbers):
                next_chapter = chapter_numbers[i + 1]
                end_token = chapter_boundaries[next_chapter] - 1
            else:
                end_token = tokens_df['token_ID_within_document'].max()
            
            # Filter tokens for this chapter
            chapter_tokens = tokens_df[
                (tokens_df['token_ID_within_document'] >= start_token) &
                (tokens_df['token_ID_within_document'] <= end_token)
            ]
            
            # Remove the chapter marker tokens themselves
            chapter_tokens = chapter_tokens[
                ~chapter_tokens['word'].str.contains(r'CHAPTER_START_\d+|\[|\]', na=False, regex=True)
            ]
            
            chapters[chapter_num] = chapter_tokens
            logger.info(f"Chapter {chapter_num}: {len(chapter_tokens)} tokens (tokens {start_token}-{end_token})")
            
        return chapters

    def extract_emotion_from_text(self, text: str) -> str:
        """Extract emotion from text using pattern matching"""
        if not isinstance(text, str):
            return 'neutral'
            
        text_lower = text.lower()
        
        emotion_scores = {}
        for emotion, patterns in self.emotion_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            emotion_scores[emotion] = score
            
        # Return emotion with highest score, default to neutral
        if max(emotion_scores.values()) > 0:
            return max(emotion_scores, key=emotion_scores.get)
        else:
            return 'neutral'

    def get_character_for_token_position(self, token_pos: int, entities_df: pd.DataFrame, characters: Dict) -> Optional[Dict]:
        """Find which character is associated with a token position"""
        # Find entities that contain this token position
        relevant_entities = entities_df[
            (entities_df['start_token'] <= token_pos) & 
            (entities_df['end_token'] >= token_pos) & 
            (entities_df['cat'] == 'PER')
        ]
        
        if relevant_entities.empty:
            return None
            
        # Get the most recent/relevant character mention
        # Prioritize proper names over pronouns
        proper_names = relevant_entities[relevant_entities['prop'] == 'PROP']
        if not proper_names.empty:
            # Use the most recent proper name
            char_id = proper_names.iloc[-1]['COREF']
        else:
            # Use the most recent pronoun
            pronouns = relevant_entities[relevant_entities['prop'] == 'PRON']
            if not pronouns.empty:
                char_id = pronouns.iloc[-1]['COREF']
            else:
                return None
                
        # Find character info
        if char_id in characters:
            return characters[char_id]
        else:
            # Check by booknlp_id
            for char_data in characters.values():
                if isinstance(char_data, dict) and char_data.get('booknlp_id') == char_id:
                    return char_data
                    
        return None

    def clean_sentence_tokens(self, sentence_tokens: pd.DataFrame) -> pd.DataFrame:
        """Clean sentence tokens by removing NaN values and invalid entries"""
        # Remove rows with NaN in the 'word' column
        clean_tokens = sentence_tokens.dropna(subset=['word'])
        
        # Convert word column to string and filter out empty strings
        clean_tokens = clean_tokens[clean_tokens['word'].astype(str).str.strip() != '']
        
        # Remove chapter markers and brackets if any slipped through
        clean_tokens = clean_tokens[
            ~clean_tokens['word'].astype(str).str.contains(r'CHAPTER_START_\d+|\[|\]', na=False, regex=True)
        ]
        
        return clean_tokens

    def create_line_from_sentence(self, sentence_tokens: pd.DataFrame, entities_df: pd.DataFrame, characters: Dict, line_counter: int) -> Optional[Dict]:
        """Create a dialogue line from a sentence with character attribution"""
        if sentence_tokens.empty:
            return None
        
        # Clean the sentence tokens first
        clean_tokens = self.clean_sentence_tokens(sentence_tokens)
        
        if clean_tokens.empty:
            return None
            
        # Get sentence text - convert all words to string and filter out NaN
        word_list = []
        for word in clean_tokens['word'].tolist():
            if pd.notna(word) and str(word).strip():
                word_list.append(str(word))
        
        if not word_list:
            return None
            
        sentence_text = ' '.join(word_list).strip()
        
        # Skip very short sentences or sentences with just punctuation
        if len(sentence_text) < 3 or sentence_text in ['!', '.', '?', ',', ';', ':']:
            return None
            
        # Find the character for this sentence
        # Use the first token position of the sentence as reference
        first_token_pos = clean_tokens.iloc[0]['token_ID_within_document']
        character_info = self.get_character_for_token_position(first_token_pos, entities_df, characters)
        
        if not character_info:
            # If no character found, try middle token
            middle_idx = len(clean_tokens) // 2
            if middle_idx < len(clean_tokens):
                middle_token_pos = clean_tokens.iloc[middle_idx]['token_ID_within_document']
                character_info = self.get_character_for_token_position(middle_token_pos, entities_df, characters)
            
        # If still no character, assign to narrator (most likely case)
        if not character_info:
            # Find narrator character
            for char_data in characters.values():
                if isinstance(char_data, dict) and char_data.get('name') == 'Narrator':
                    character_info = char_data
                    break
                    
        # If still no character, create a default narrator entry
        if not character_info:
            character_info = {
                'character_id': 'char_narrator',
                'name': 'Narrator',
                'booknlp_id': 0,
                'age_group': 'Young Adult',
                'gender': 'Male',
                'personality': {'O': 0.75, 'C': 0.55, 'E': 0.35, 'A': 0.50, 'N': 0.70}
            }
            
        # Extract emotion
        emotion = self.extract_emotion_from_text(sentence_text)
        
        # Create line entry
        line_entry = {
            'line_id': f"line_{line_counter:06d}",
            'character_id': character_info['character_id'],
            'character_name': character_info['name'],
            'text': sentence_text,
            'emotion': emotion,
            'start_token': int(clean_tokens.iloc[0]['token_ID_within_document']),
            'end_token': int(clean_tokens.iloc[-1]['token_ID_within_document']),
            'sentence_id': int(clean_tokens.iloc[0]['sentence_ID']),
            'paragraph_id': int(clean_tokens.iloc[0]['paragraph_ID'])
        }
        
        return line_entry

    def process_chapter_lines(self, chapter_tokens: pd.DataFrame, entities_df: pd.DataFrame, characters: Dict, chapter_num: int) -> List[Dict]:
        """Process all sentences in a chapter as lines with character attribution"""
        logger.info(f"Processing chapter {chapter_num} sentences as dialogue lines...")
        
        dialogue_lines = []
        line_counter = 1
        
        # Group tokens by sentence within this chapter
        for sentence_id, sentence_tokens in chapter_tokens.groupby('sentence_ID'):
            line_entry = self.create_line_from_sentence(sentence_tokens, entities_df, characters, line_counter)
            
            if line_entry:
                # Add chapter info to line entry
                line_entry['chapter'] = chapter_num
                dialogue_lines.append(line_entry)
                line_counter += 1
                
        logger.info(f"Created {len(dialogue_lines)} dialogue lines from chapter {chapter_num}")
        return dialogue_lines

    def create_chapter_json(self, chapter_num: int, lines: List[Dict]) -> Dict:
        """Create chapter JSON structure"""
        character_counts = defaultdict(int)
        emotion_counts = defaultdict(int)
        
        for line in lines:
            character_counts[line['character_id']] += 1
            emotion_counts[line['emotion']] += 1
        
        return {
            'chapter': chapter_num,
            'total_lines': len(lines),
            'characters_in_chapter': list(character_counts.keys()),
            'character_line_counts': dict(character_counts),
            'emotion_distribution': dict(emotion_counts),
            'lines': lines
        }

    def get_chapter_filename(self, book_id: str, chapter_num: int) -> str:
        """Generate chapter filename with book_id prefix"""
        return f"{book_id}_chapter{chapter_num:04d}.json"
    
    def chapter_exists(self, book_id: str, chapter_num: int) -> bool:
        """Check if chapter file already exists"""
        filename = self.get_chapter_filename(book_id, chapter_num)
        filepath = self.dialogue_dir / filename
        return filepath.exists()

    def save_chapter_dialogue(self, book_id: str, chapter_num: int, lines: List[Dict]) -> Optional[str]:
        """Save dialogue for a single chapter to JSON file (skip if exists)"""
        # Check if chapter already exists
        if self.chapter_exists(book_id, chapter_num):
            filename = self.get_chapter_filename(book_id, chapter_num)
            logger.info(f"📁 Chapter {chapter_num} already exists, skipping: {filename}")
            return str(self.dialogue_dir / filename)
        
        # Create chapter data
        chapter_data = self.create_chapter_json(chapter_num, lines)
        
        # Generate filename with book_id prefix
        filename = self.get_chapter_filename(book_id, chapter_num)
        filepath = self.dialogue_dir / filename
        
        # Custom JSON encoder to handle numpy int64
        def json_serial(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chapter_data, f, indent=2, ensure_ascii=False, default=json_serial)
            
        logger.info(f"💾 Created chapter {chapter_num}: {len(lines)} lines to {filename}")
        return str(filepath)

    def create_dialogue_summary(self, all_chapters_info: List[Dict]) -> Dict:
        """Create summary of dialogue processing"""
        total_lines = sum(info['total_lines'] for info in all_chapters_info)
        total_chapters = len(all_chapters_info)
        all_characters = set()
        emotion_counts = defaultdict(int)
        
        for chapter_info in all_chapters_info:
            for char_id in chapter_info['characters_in_chapter']:
                all_characters.add(char_id)
            for emotion, count in chapter_info['emotion_distribution'].items():
                emotion_counts[emotion] += count
                
        return {
            'total_chapters': total_chapters,
            'total_dialogue_lines': total_lines,
            'unique_characters': len(all_characters),
            'emotion_distribution': dict(emotion_counts),
            'characters_in_dialogue': list(all_characters),
            'chapters_processed': [info['chapter'] for info in all_chapters_info]
        }

    def process_book_dialogue(self, book_id: str, characters_file: str = "data/processed/characters/advanced_characters.json") -> Tuple[Dict, List[str]]:
        """Main dialogue processing pipeline - processes ALL lines with chapter separation"""
        logger.info(f"Starting chapter-wise dialogue processing for book: {book_id}")
        
        # Load characters and BookNLP data
        characters = self.load_characters(characters_file)
        data = self.load_booknlp_data(book_id)
        
        if 'tokens' not in data:
            raise ValueError(f"No tokens data found for {book_id}")
        
        # Detect chapter boundaries
        chapter_boundaries = self.detect_chapter_boundaries(data['tokens'])
        
        # Assign tokens to chapters
        chapter_tokens = self.assign_tokens_to_chapters(data['tokens'], chapter_boundaries)
        
        # Check existing chapters and process only missing ones
        existing_chapters = []
        missing_chapters = []
        
        for chapter_num in sorted(chapter_tokens.keys()):
            if self.chapter_exists(book_id, chapter_num):
                existing_chapters.append(chapter_num)
            else:
                missing_chapters.append(chapter_num)
        
        if existing_chapters:
            logger.info(f"📁 Found {len(existing_chapters)} existing chapters: {existing_chapters}")
        
        if missing_chapters:
            logger.info(f"🔨 Processing {len(missing_chapters)} missing chapters: {missing_chapters}")
        else:
            logger.info("✅ All chapters already exist, no processing needed")
        
        # Process each missing chapter
        saved_files = []
        all_chapters_info = []
        
        for chapter_num in sorted(chapter_tokens.keys()):
            if chapter_num in missing_chapters:
                logger.info(f"Processing chapter {chapter_num}...")
                
                # Process lines for this chapter
                chapter_lines = self.process_chapter_lines(
                    chapter_tokens[chapter_num], 
                    data['entities'], 
                    characters, 
                    chapter_num
                )
                
                if chapter_lines:
                    # Save chapter dialogue
                    filepath = self.save_chapter_dialogue(book_id, chapter_num, chapter_lines)
                    if filepath:
                        saved_files.append(filepath)
                    
                    # Store chapter info for summary
                    chapter_info = self.create_chapter_json(chapter_num, chapter_lines)
                    all_chapters_info.append(chapter_info)
                else:
                    logger.warning(f"No dialogue lines found for chapter {chapter_num}")
            
            else:
                # Chapter exists, just add to files list for summary
                existing_filepath = str(self.dialogue_dir / self.get_chapter_filename(book_id, chapter_num))
                saved_files.append(existing_filepath)
                
                # Try to load existing chapter info for summary
                try:
                    with open(existing_filepath, 'r', encoding='utf-8') as f:
                        existing_chapter_data = json.load(f)
                        all_chapters_info.append(existing_chapter_data)
                except Exception as e:
                    logger.warning(f"Could not load existing chapter {chapter_num}: {e}")
                    # Create minimal info
                    all_chapters_info.append({
                        'chapter': chapter_num,
                        'total_lines': 0,
                        'characters_in_chapter': [],
                        'character_line_counts': {},
                        'emotion_distribution': {}
                    })
        
        # Create summary
        summary = self.create_dialogue_summary(all_chapters_info)
        summary['newly_processed_chapters'] = len(missing_chapters)
        summary['existing_chapters'] = len(existing_chapters)
        summary['total_chapter_files'] = len(saved_files)
        
        logger.info(f"Completed chapter-wise dialogue processing for {book_id}")
        logger.info(f"📊 Summary: {len(missing_chapters)} new + {len(existing_chapters)} existing = {len(saved_files)} total chapter files")
        return summary, saved_files

def main():
    """Example usage"""
    processor = FixedDialogueProcessor()
    
    try:
        summary, files = processor.process_book_dialogue("samples")
        
        print(f"\n✅ Chapter-wise dialogue processing complete!")
        print(f"📁 Total chapter files: {len(files)}")
        print(f"📊 Summary:")
        print(f"   Chapters: {summary['total_chapters']}")
        print(f"   Total lines: {summary['total_dialogue_lines']}")
        print(f"   Characters: {summary['unique_characters']}")
        print(f"   Newly processed: {summary.get('newly_processed_chapters', 0)}")
        print(f"   Already existed: {summary.get('existing_chapters', 0)}")
        print(f"   Chapters processed: {summary['chapters_processed']}")
        print(f"   Emotion distribution: {summary['emotion_distribution']}")
        
        # Show sample from first few chapters
        sample_files = [f for f in files if Path(f).exists()][:3]  # First 3 existing files
        for i, filepath in enumerate(sample_files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    chapter_data = json.load(f)
                    print(f"\n📖 Chapter {chapter_data['chapter']} ({Path(filepath).name}):")
                    print(f"   Lines: {chapter_data['total_lines']}")
                    print(f"   Characters: {len(chapter_data['characters_in_chapter'])}")
                    
                    # Show first 2 lines
                    for line in chapter_data['lines'][:2]:
                        print(f"   {line['character_name']}: \"{line['text'][:50]}...\" ({line['emotion']})")
            except Exception as e:
                print(f"   Could not load chapter data: {e}")
                    
    except Exception as e:
        logger.error(f"Chapter-wise dialogue processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
