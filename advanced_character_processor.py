#!/usr/bin/env python3
"""
Advanced Character Processing System for Novel Analysis
Sophisticated character identification, dialogue attribution, and personality analysis
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
from difflib import SequenceMatcher
import warnings
import os
import random
import google.generativeai as genai
import time

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GeminiCharacterEnhancer:
    """
    Class to handle Gemini API integration for character enhancement
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY') or 'AIzaSyD23o0K1sI4lghuNJhnEnUMu_B-jaUf9i4'
        self.model = None
        self.rate_limit_delay = float(os.getenv('GEMINI_RATE_LIMIT_DELAY', '1'))  # seconds between API calls
        self.enhancement_enabled = os.getenv('GEMINI_CHARACTER_ENHANCEMENT', 'True').lower() == 'true'
        self.enhancement_threshold = float(os.getenv('GEMINI_ENHANCEMENT_THRESHOLD', '0.3'))
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
                self.model = None
        else:
            logger.warning("No Gemini API key provided. Character enhancement will use fallback methods.")
    
    def enhance_character_data(self, character_data: Dict) -> Dict:
        """
        Use Gemini to enhance and refine character data
        """
        if not self.model or not self.enhancement_enabled:
            return character_data
        
        try:
            # Prepare context for Gemini
            context = self._prepare_character_context(character_data)
            
            prompt = f"""
            You are a literary character analyst. Based on the following character information from a novel, 
            please analyze and enhance the character profile. Provide refined insights about their personality, 
            background, and key traits.
            
            Character Information:
            {context}
            
            Please provide a JSON response with the following enhanced fields:
            - refined_personality: Enhanced personality analysis with specific traits
            - character_background: Likely background and history
            - key_relationships: Important relationships inferred from the text
            - character_arc: Brief description of their role/arc in the story
            - dialogue_style: How they typically speak or express themselves
            - enhanced_traits: Additional character traits not captured in basic analysis
            
            Keep the response concise but insightful. Focus on literary analysis.
            """
            
            response = self.model.generate_content(prompt)
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            # Try to parse JSON response
            try:
                enhanced_data = json.loads(response.text)
                character_data.update({'gemini_enhanced': enhanced_data})
                logger.info(f"Enhanced character data for {character_data.get('name', 'unknown')}")
            except json.JSONDecodeError:
                # Fallback: store as text analysis
                character_data['gemini_enhanced'] = {'analysis': response.text}
                
        except Exception as e:
            logger.error(f"Error enhancing character with Gemini: {e}")
        
        return character_data
    
    def generate_detailed_physical_description(self, character_data: Dict) -> str:
        """
        Use Gemini to generate a detailed, creative physical description
        """
        if not self.model or not self.enhancement_enabled:
            return self._fallback_physical_description(character_data)
        
        try:
            context = self._prepare_character_context(character_data)
            existing_description = character_data.get('physical_description', '')
            
            prompt = f"""
            You are a creative writer specializing in character descriptions. Based on the following character 
            information from a novel, create a vivid, detailed physical description that would fit in a 
            high-quality literary work.
            
            Character Information:
            {context}
            
            Current basic description: {existing_description}
            
            Please create a detailed, evocative physical description (2-3 paragraphs) that:
            - Captures the character's essence and personality through their appearance
            - Uses vivid, literary language
            - Includes specific details about facial features, build, mannerisms, and presence
            - Reflects their age, personality traits, and role in the story
            - Avoids clichés and generic descriptions
            - Creates a memorable, unique visual impression
            
            Write in a literary style that would fit in a published novel.
            """
            
            response = self.model.generate_content(prompt)
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            enhanced_description = response.text.strip()
            logger.info(f"Generated enhanced physical description for {character_data.get('name', 'unknown')}")
            return enhanced_description
            
        except Exception as e:
            logger.error(f"Error generating physical description with Gemini: {e}")
            return self._fallback_physical_description(character_data)
    
    def _prepare_character_context(self, character_data: Dict) -> str:
        """
        Prepare character context for Gemini prompts
        """
        context_parts = [
            f"Name: {character_data.get('name', 'Unknown')}",
            f"Gender: {character_data.get('gender', 'Unknown')}",
            f"Age Group: {character_data.get('age_group', 'Unknown')}",
            f"Mentions: {character_data.get('mention_count', 0)}"
        ]
        
        # Add personality if available
        if 'personality' in character_data:
            personality = character_data['personality']
            context_parts.append(f"Personality Traits: Openness={personality.get('openness', 0.5):.2f}, Conscientiousness={personality.get('conscientiousness', 0.5):.2f}, Extraversion={personality.get('extraversion', 0.5):.2f}, Agreeableness={personality.get('agreeableness', 0.5):.2f}, Neuroticism={personality.get('neuroticism', 0.5):.2f}")
        
        # Add sample quotes
        if 'quotes' in character_data:
            quotes = [q.get('quote_text', '') if isinstance(q, dict) else str(q) for q in character_data['quotes'][:3]]
            context_parts.append(f"Sample Dialogue: {' | '.join(quotes)}")
        
        # Add context texts
        if 'context_texts' in character_data:
            context_texts = character_data['context_texts'][:2]  # Limit to avoid token limits
            context_parts.append(f"Context: {' '.join(context_texts)}")
        
        return '\n'.join(context_parts)
    
    def _fallback_physical_description(self, character_data: Dict) -> str:
        """
        Fallback method for generating physical descriptions without Gemini
        """
        return character_data.get('physical_description', 'A character with distinctive features and presence.')


class AdvancedCharacterProcessor:
    """
    Advanced character processing system that addresses:
    1. Pronoun character removal and reassignment
    2. Misattributed quote correction
    3. Character merging and consolidation
    4. Sophisticated quote attribution
    5. Personality analysis using pre-trained models
    6. Improved gender classification
    7. Gemini AI-powered character enhancement and physical descriptions
    """

    def __init__(self, output_dir: str = "modelsbooknlp/output", characters_dir: str = "data/processed/characters", gemini_api_key: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.characters_dir = Path(characters_dir)
        self.characters_dir.mkdir(exist_ok=True)

        # Character storage
        self.characters = {}
        self.character_mentions = defaultdict(list)
        self.character_quotes = defaultdict(list)
        self.narrator_mentions = defaultdict(list)

        # Patterns for filtering - comprehensive pronoun and generic word removal
        self.PRONOUN_PATTERNS = [
            # Personal pronouns (1st, 2nd, 3rd person)
            r'\b(i|me|my|mine|myself)\b',  # 1st person singular
            r'\b(we|us|our|ours|ourselves)\b',  # 1st person plural
            r'\b(you|your|yours|yourself|yourselves)\b',  # 2nd person
            r'\b(he|him|his|himself)\b',  # 3rd person masculine
            r'\b(she|her|hers|herself)\b',  # 3rd person feminine
            r'\b(it|its|itself)\b',  # 3rd person neuter
            r'\b(they|them|their|theirs|themselves)\b',  # 3rd person plural
            # Demonstrative pronouns
            r'\b(this|that|these|those)\b',
            # Relative pronouns
            r'\b(who|whom|whose|which|what|where|when|why|how)\b',
            # Indefinite pronouns
            r'\b(one|oneself|anyone|everyone|someone|no one|nobody|somebody|anybody)\b'
        ]

        self.EXCLUDE_PATTERNS = [
            # Narrative and generic terms
            r'\b(narrator|author|reader|character|protagonist|antagonist)\b',
            # Chapter markers
            r'\[CHAPTER_START_\d+\]',
            # Articles with generic nouns
            r'\b(the|a|an)\s+\w+',
            # Generic descriptors that shouldn't be characters
            r'\b(man|woman|person|people|individual|human|being)\b',
            r'\b(child|children|kid|kids|boy|girl|baby|infant)\b',
            r'\b(guy|guys|dude|fellow|lady|ladies|gentleman|gentlemen)\b',
            r'\b(father|mother|parent|parents|son|daughter|brother|sister)\b',
            r'\b(friend|friends|enemy|enemies|stranger|strangers)\b',
            # Common single letters that get misidentified
            r'^[a-zA-Z]$',
            # Numbers
            r'^\d+$',
            # Very short words (2 chars or less) that are likely not names
            r'^.{1,2}$'
        ]

        # Dialogue attribution patterns
        self.DIALOGUE_TAGS = [
            'said', 'asked', 'replied', 'answered', 'exclaimed', 'shouted', 'whispered',
            'murmured', 'muttered', 'cried', 'laughed', 'sighed', 'gasped', 'groaned',
            'moaned', 'yelled', 'screamed', 'barked', 'snapped', 'growled', 'hissed',
            'spoke', 'told', 'announced', 'declared', 'stated', 'mentioned', 'added',
            'continued', 'interrupted', 'concluded', 'finished', 'began', 'started'
        ]

        self.gender_detector = None
        self.personality_classifier = None
        self.SIMILARITY_THRESHOLD = 0.8
        self.MIN_MENTIONS_FOR_CHARACTER = 2
        
        # Initialize Gemini enhancer
        self.gemini_enhancer = GeminiCharacterEnhancer(api_key=gemini_api_key)

    # ------------------ Loading BookNLP output ------------------

    def load_booknlp_output(self, book_id: str) -> Dict[str, pd.DataFrame]:
        """Load BookNLP output files into DataFrames"""
        logger.info(f"Loading BookNLP output for book_id: {book_id}")

        files = {
            'entities': f"{book_id}.entities",
            'quotes': f"{book_id}.quotes",
            'tokens': f"{book_id}.tokens"
        }

        data = {}
        for key, filename in files.items():
            filepath = self.output_dir / filename
            if filepath.exists():
                try:
                    if key == 'tokens':
                        data[key] = pd.read_csv(filepath, sep='\t', quoting=3, on_bad_lines='skip')
                    else:
                        data[key] = pd.read_csv(filepath, sep='\t', on_bad_lines='skip', quoting=3)
                    logger.info(f"Loaded {key}: {len(data[key])} entries")
                except Exception as e:
                    logger.error(f"Error loading {key}: {e}")
            else:
                logger.warning(f"File not found: {filepath}")

        return data

    # ------------------ Character Processing ------------------

    def extract_chapter_boundaries(self, tokens_df: pd.DataFrame) -> List[Tuple[int, int]]:
        """Extract chapter boundaries using [CHAPTER_START_{number}] markers"""
        chapter_boundaries = []
        current_chapter_start = 0

        for idx, row in tokens_df.iterrows():
            if '[CHAPTER_START_' in str(row.get('word', '')):
                if current_chapter_start > 0:
                    chapter_boundaries.append((current_chapter_start, idx - 1))
                current_chapter_start = idx

        if current_chapter_start < len(tokens_df):
            chapter_boundaries.append((current_chapter_start, len(tokens_df) - 1))

        logger.info(f"Found {len(chapter_boundaries)} chapters")
        return chapter_boundaries

    def is_pronoun_character(self, character_name: str) -> bool:
        """Check if a character is a pronoun that should be filtered out"""
        name_lower = character_name.lower().strip()

        # First check pronoun patterns
        for pattern in self.PRONOUN_PATTERNS:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return True

        # Then check exclude patterns
        for pattern in self.EXCLUDE_PATTERNS:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return True

        return False
   
    def should_merge_characters(self, char1_name: str, char2_name: str) -> bool:
        """Determine if two characters should be merged based on name similarity - improved precision"""
        from difflib import SequenceMatcher
        
        # If names are identical, they're the same
        if char1_name.lower() == char2_name.lower():
            return True
            
        # Don't merge if both are proper names with different core names
        # e.g., "Mr.A" and "Mr.Z" should NOT be merged
        if self.is_proper_name(char1_name) and self.is_proper_name(char2_name):
            # Extract core names (remove titles, prefixes, suffixes)
            core1 = self.extract_core_name(char1_name)
            core2 = self.extract_core_name(char2_name)
            
            # If core names are different, don't merge
            if core1.lower() != core2.lower():
                return False
                
        # Check for substring relationships (one name contains the other)
        if char1_name.lower() in char2_name.lower() or char2_name.lower() in char1_name.lower():
            # But be careful - "Mr.A" should not merge with "Mr.Alex"
            if len(char1_name) > 3 and len(char2_name) > 3:  # Both are substantial names
                return True
            elif len(char1_name) <= 3 or len(char2_name) <= 3:  # One is very short
                return False
                
        # Use similarity threshold for other cases
        similarity = SequenceMatcher(None, char1_name.lower(), char2_name.lower()).ratio()
        return similarity >= self.SIMILARITY_THRESHOLD

    def extract_core_name(self, name: str) -> str:
        """Extract the core name from a character name, removing titles and prefixes"""
        name = name.strip()
        
        # Common titles to remove
        titles = ['mr.', 'mrs.', 'miss', 'ms.', 'dr.', 'prof.', 'professor', 'sir', 'madam', 'lord', 'lady']
        
        # Remove titles from the beginning
        for title in titles:
            if name.lower().startswith(title.lower()):
                name = name[len(title):].strip()
                break
                
        # Remove common suffixes
        suffixes = ['jr.', 'sr.', 'ii', 'iii', 'iv', 'v']
        for suffix in suffixes:
            if name.lower().endswith('.' + suffix.lower()):
                name = name[:-len(suffix)-1].strip()
                break
                
        return name

    def is_pronoun_character(self, name: str) -> bool:
        """Check if a character name is a pronoun that should be removed"""
        name_lower = name.lower().strip()
        
        # Common pronouns that should not be characters
        pronouns = {
            'he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'their', 'theirs',
            'it', 'its', 'this', 'that', 'these', 'those', 'who', 'whom', 'which', 'what',
            'where', 'when', 'why', 'how', 'oneself', 'myself', 'yourself', 'himself', 
            'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 'yours', 'mine',
            'ours', 'yourselves', 'themselves'
        }
        
        # Check for common pronoun patterns
        pronoun_patterns = [
            r'^[a-z]+$',  # All lowercase single words
            r'^[A-Z][a-z]+$'  # Capitalized single words that are pronouns
        ]
        
        if name_lower in pronouns:
            return True
            
        # Check if it's a generic descriptor that might be a pronoun
        generic_descriptors = ['monkeys', 'people', 'person', 'man', 'woman', 'child', 'boy', 'girl']
        if name_lower in generic_descriptors:
            return True
            
        return False

    def is_proper_name(self, name: str) -> bool:
        """Check if a character has a proper name (not a pronoun or generic descriptor)"""
        name_lower = name.lower().strip()
        
        # If it's a pronoun, it's not a proper name
        if self.is_pronoun_character(name):
            return False
            
        # Check for proper name patterns
        # Names should be capitalized and not be common words
        if not name[0].isupper():
            return False
            
        # Check if it's a common word that shouldn't be a character name
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'among', 'against', 'within', 'without', 'upon', 'across', 'behind', 'beyond',
            'under', 'over', 'around', 'near', 'far', 'here', 'there', 'where', 'when', 'why', 'how',
            'what', 'who', 'which', 'that', 'this', 'these', 'those', 'some', 'any', 'all', 'every',
            'each', 'both', 'either', 'neither', 'one', 'two', 'three', 'first', 'second', 'last',
            'next', 'other', 'another', 'same', 'different', 'new', 'old', 'young', 'big', 'small',
            'good', 'bad', 'right', 'wrong', 'true', 'false', 'yes', 'no', 'maybe', 'perhaps',
            'always', 'never', 'sometimes', 'often', 'usually', 'rarely', 'hardly', 'almost',
            'quite', 'very', 'rather', 'pretty', 'really', 'actually', 'finally', 'suddenly',
            'immediately', 'quickly', 'slowly', 'carefully', 'easily', 'hardly', 'nearly',
            'almost', 'exactly', 'precisely', 'certainly', 'definitely', 'probably', 'possibly',
            'perhaps', 'maybe', 'surely', 'obviously', 'clearly', 'apparently', 'evidently',
            'supposedly', 'allegedly', 'reportedly', 'apparently', 'evidently', 'obviously',
            'clearly', 'surely', 'certainly', 'definitely', 'probably', 'possibly', 'perhaps',
            'maybe', 'likely', 'unlikely', 'impossible', 'possible', 'probable', 'improbable',
            'yours', 'mine', 'ours', 'theirs'
        }
        
        if name_lower in common_words:
            return False
            
        # Check if it's a Character_XXX pattern (unnamed character)
        if name.startswith('Character_'):
            return False
            
        return True

    def consolidate_characters(self, characters: Dict) -> Dict:
        """Merge similar characters and remove pronouns"""
        logger.info("Consolidating characters...")

        filtered_characters = {}
        pronoun_characters = {}

        for char_id, char_data in characters.items():
            # Use the comprehensive validation method
            if not self.is_valid_character_name(char_data['name']):
                pronoun_characters[char_id] = char_data
                logger.info(f"Filtered out invalid character: {char_data['name']}")
            else:
                filtered_characters[char_id] = char_data

        consolidated = {}
        processed_ids = set()

        for char_id, char_data in filtered_characters.items():
            if char_id in processed_ids:
                continue

            merged_data = char_data.copy()
            merged_data['merged_from'] = [char_data['name']]
            merged_data['merged_ids'] = [char_id]

            for other_id, other_data in filtered_characters.items():
                if other_id != char_id and other_id not in processed_ids and self.should_merge_characters(char_data['name'], other_data['name']):
                    merged_data['mention_count'] += other_data['mention_count']
                    merged_data['context_texts'].extend(other_data['context_texts'])
                    merged_data['all_mentions'].extend(other_data['all_mentions'])
                    merged_data['merged_from'].append(other_data['name'])
                    merged_data['merged_ids'].append(other_id)
                    if len(other_data['name']) > len(merged_data['name']):
                        merged_data['name'] = other_data['name']
                    processed_ids.add(other_id)
                    logger.info(f"Merged '{other_data['name']}' into '{merged_data['name']}'")

            consolidated[char_id] = merged_data
            processed_ids.add(char_id)

        self.pronoun_characters = pronoun_characters
        logger.info(f"Consolidated {len(characters)} characters into {len(consolidated)}")
        return consolidated

    # ------------------ Quote Attribution ------------------
    
    def analyze_quote_context(self, quote_row: pd.DataFrame, tokens_df: pd.DataFrame, characters: Dict) -> Dict:
        """Analyze the context around a quote to determine attribution confidence"""
        start_token = quote_row['quote_start']
        end_token = quote_row['quote_end']
        mention_phrase = quote_row['mention_phrase']

        # Get surrounding context (50 tokens before and after)
        context_start = max(0, start_token - 50)
        context_end = min(len(tokens_df), end_token + 50)

        context_tokens = tokens_df.iloc[context_start:context_end]['word'].tolist()
        context_tokens = [str(token) for token in context_tokens if pd.notna(token)]
        context_text = ' '.join(context_tokens).lower()

        # Look for dialogue tags
        dialogue_tags_found = []
        for tag in self.DIALOGUE_TAGS:
            if tag in context_text:
                dialogue_tags_found.append(tag)

        # Analyze quote structure
        quote_text = quote_row['quote']
        has_quotes = quote_text.startswith('"') and quote_text.endswith('"')

        # Determine mention type
        mention_type = 'unknown'
        if any(tag in context_text for tag in self.DIALOGUE_TAGS):
            mention_type = 'dialogue_tag'
        elif 'thought' in context_text or 'think' in context_text:
            mention_type = 'internal_thought'
        elif any(word in context_text for word in ['looked', 'glanced', 'stared', 'gazed']):
            mention_type = 'physical_description'
        elif any(word in context_text for word in ['felt', 'sensed', 'realized', 'understood']):
            mention_type = 'emotional_description'

        # Calculate confidence
        confidence = 0.5
        if dialogue_tags_found:
            confidence += 0.3
        if has_quotes:
            confidence += 0.2
        if mention_type == 'dialogue_tag':
            confidence += 0.2
        elif mention_type in ['internal_thought', 'physical_description']:
            confidence -= 0.3

        confidence = max(0.0, min(1.0, confidence))

        return {
            'confidence': confidence,
            'mention_type': mention_type,
            'context': context_text,
            'dialogue_tags': dialogue_tags_found,
            'has_quotes': has_quotes
        }

    def is_actual_dialogue(self, quote_text: str, mention_phrase: str, context_analysis: dict) -> bool:
        """
        Determine whether a quote is actual dialogue or narration/thought.
        """
        # Basic rules
        if not quote_text or len(quote_text.strip()) == 0:
            return False
        
        # Quotes with quotation marks are likely dialogue
        if quote_text.startswith('"') and quote_text.endswith('"'):
            return True
        
        # If context mentions a dialogue tag, consider it dialogue
        if context_analysis.get('mention_type') == 'dialogue_tag':
            return True
        
        # Otherwise, treat as non-dialogue (internal thought / narration)
        return False


    def fix_quote_attribution(self, quotes_df: pd.DataFrame, characters: Dict, tokens_df: pd.DataFrame) -> Dict[str, List]:
        """Fix quote attribution using sophisticated analysis"""
        logger.info("Fixing quote attribution...")

        character_quotes = defaultdict(list)
        narrator_mentions = defaultdict(list)

        for _, quote_row in quotes_df.iterrows():
            char_id = quote_row['char_id']
            quote_text = quote_row['quote']
            mention_phrase = quote_row['mention_phrase']

            if char_id not in characters:
                continue

            character = characters[char_id]
            character_id = character['character_id']

            context_analysis = self.analyze_quote_context(
                quote_row, tokens_df, characters
            )

            if self.is_actual_dialogue(quote_text, mention_phrase, context_analysis):
                quote_data = {
                    'quote_text': quote_text,
                    'start_token': quote_row['quote_start'],
                    'end_token': quote_row['quote_end'],
                    'character_name': mention_phrase,
                    'speaker_confidence': context_analysis['confidence'],
                    'context_type': 'dialogue',
                    'dialogue_analysis': context_analysis
                }
                character_quotes[character_id].append(quote_data)
            else:
                narrator_data = {
                    'mention_text': f"{mention_phrase}: {quote_text}",
                    'mention_type': context_analysis['mention_type'],
                    'context': context_analysis['context'],
                    'start_token': quote_row['quote_start'],
                    'end_token': quote_row['quote_end']
                }
                narrator_mentions[character_id].append(narrator_data)

        # Ensure at least 5 quotes per character
        for char_id, char_data in characters.items():
            character_id = char_data['character_id']
            if len(character_quotes[character_id]) < 5:
                placeholder_quotes = self.generate_placeholder_quotes(char_data, 5 - len(character_quotes[character_id]))
                character_quotes[character_id].extend(placeholder_quotes)

        self.narrator_mentions = narrator_mentions

        logger.info(f"Processed quotes for {len(character_quotes)} characters")
        logger.info(f"Identified {sum(len(mentions) for mentions in narrator_mentions.values())} narrator mentions")

        return character_quotes


        

    # ------------------ Analysis Helpers ------------------

    # (Methods: generate_placeholder_quotes, analyze_quote_context, is_actual_dialogue,
    # classify_gender_advanced, classify_age, analyze_personality_with_models, etc.)
    # ... [all the helper methods from your original code remain unchanged] ...

    # ------------------ Final Processing ------------------

    def generate_placeholder_quotes(self, char_data: dict, n: int) -> list:
        """
        Generate placeholder quotes for a character if they have less than the minimum required.
        """
        placeholder_quotes = []
        for i in range(n):
            placeholder_quotes.append({
                'quote_text': f"[Placeholder quote {i+1} for {char_data['name']}]",
                'start_token': None,
                'end_token': None,
                'character_name': char_data['name'],
                'speaker_confidence': 0.0,
                'context_type': 'placeholder',
                'dialogue_analysis': {}
            })
        return placeholder_quotes


    def create_character_entries(self, characters: Dict, character_quotes: Dict) -> List[Dict]:
        """Create final character entries with all advanced analysis including Gemini enhancement"""
        character_list = []
        
        for char_id, char_data in characters.items():
            if char_data['mention_count'] < self.MIN_MENTIONS_FOR_CHARACTER:
                continue

            quotes = character_quotes.get(char_id, [])
            narrator_mentions = self.narrator_mentions.get(char_id, [])

            gender = self.classify_gender_advanced(
                char_data['name'], 
                char_data['context_texts'], 
                quotes
            )

            age_group = self.classify_age(
                char_data['name'], 
                char_data['context_texts'], 
                quotes
            )

            personality = self.analyze_personality_with_models(quotes, char_data['context_texts'])
            
            # Generate basic character looks
            character_looks = self.generate_character_looks(
                char_data['name'], 
                gender, 
                age_group, 
                personality, 
                quotes, 
                char_data['context_texts']
            )

            importance_score = min(1.0, char_data['mention_count'] / 100.0)
            personality_confidence = min(1.0, len(quotes) / 10.0 + len(char_data['context_texts']) / 20.0)

            # Create initial character entry
            character_entry = {
                'character_id': char_data['character_id'],
                'name': char_data['name'],
                'booknlp_id': char_data['booknlp_id'],
                'gender': gender.title(),
                'age_group': age_group.replace('_', ' ').title(),
                'personality': personality,
                'quotes': quotes[:5],
                'narrator_mentions': narrator_mentions[:3],
                'context_texts': char_data['context_texts'][:3],  # Add for Gemini
                'mention_count': char_data['mention_count'],
                'is_proper_name': char_data['is_proper_name'],
                'physical_description': character_looks.get('physical_description', ''),
                'character_analysis': {
                    'total_dialogue_lines': len(quotes),
                    'total_narrator_mentions': len(narrator_mentions),
                    'personality_analysis_confidence': round(personality_confidence, 2),
                    'gender_classification_method': 'advanced_multi_method',
                    'merged_from': char_data.get('merged_from', [char_data['name']]),
                    'importance_score': round(importance_score, 2)
                },
                'audio_file': f"audio/{char_data['name'].replace(' ', '_')}/"
            }
            
            # Add detailed character looks
            character_entry.update(character_looks)
            
            # Enhance with Gemini API (only for important characters to manage API usage)
            threshold = self.gemini_enhancer.enhancement_threshold
            if importance_score > threshold and self.gemini_enhancer.enhancement_enabled:
                logger.info(f"Enhancing character {char_data['name']} with Gemini AI (importance: {importance_score:.2f} > {threshold:.2f})...")
                
                # Enhance character data
                enhanced_entry = self.gemini_enhancer.enhance_character_data(character_entry)
                
                # Generate enhanced physical description
                enhanced_physical_description = self.gemini_enhancer.generate_detailed_physical_description(enhanced_entry)
                enhanced_entry['physical_description_enhanced'] = enhanced_physical_description
                enhanced_entry['physical_description_original'] = character_looks.get('physical_description', '')
                
                # Use enhanced description as primary
                enhanced_entry['physical_description'] = enhanced_physical_description
                
                character_entry = enhanced_entry
            else:
                if self.gemini_enhancer.enhancement_enabled:
                    logger.info(f"Skipping Gemini enhancement for {char_data['name']} (importance: {importance_score:.2f} <= {threshold:.2f})")
                else:
                    logger.info(f"Gemini enhancement disabled for {char_data['name']}")

            character_list.append(character_entry)

        character_list.sort(key=lambda x: x['mention_count'], reverse=True)
        logger.info(f"Created {len(character_list)} advanced character entries with Gemini enhancement")
        return character_list
        return character_list

    def analyze_name_gender(self, name: str) -> str:
        """Analyze gender based on name patterns"""
        name_lower = name.lower().strip()
        
        # Male name patterns
        male_names = ['john', 'michael', 'david', 'james', 'robert', 'william', 'richard', 'charles', 'thomas', 'christopher',
                     'daniel', 'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua', 'kenneth',
                     'kevin', 'brian', 'george', 'timothy', 'ronald', 'jason', 'edward', 'jeffrey', 'ryan', 'jacob',
                     'gary', 'nicholas', 'eric', 'jonathan', 'stephen', 'larry', 'justin', 'scott', 'brandon', 'benjamin',
                     'samuel', 'frank', 'gregory', 'raymond', 'alexander', 'patrick', 'jack', 'dennis', 'jerry', 'tyler',
                     'aaron', 'jose', 'henry', 'douglas', 'adam', 'peter', 'nathan', 'zachary', 'kyle', 'walter',
                     'harold', 'jeremy', 'ethan', 'carl', 'keith', 'roger', 'gerald', 'arthur', 'sean', 'christian',
                     'terry', 'lawrence', 'austin', 'joe', 'noah', 'jesse', 'albert', 'bryan', 'billy', 'bruce',
                     'wayne', 'eugene', 'louis', 'philip', 'bobby', 'johnny', 'roy', 'ralph', 'eugene', 'howard',
                     'dunn', 'leonard', 'benson', 'derrick', 'megose', 'kenley', 'lanevus', 'daxter', 'glacis', 'bogda',
                     'fors', 'crestet', 'welch', 'alger', 'qilangos', 'bitsch', 'mountbatten', 'dad', 'thief', 'dun']
        
        # Female name patterns  
        female_names = ['mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen',
                       'nancy', 'lisa', 'betty', 'helen', 'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle',
                       'laura', 'sarah', 'kimberly', 'deborah', 'dorothy', 'lisa', 'nancy', 'karen', 'betty', 'helen',
                       'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle', 'laura', 'sarah', 'kimberly', 'deborah',
                       'dorothy', 'amy', 'angela', 'brenda', 'emma', 'olivia', 'cynthia', 'marie', 'janet', 'catherine',
                       'frances', 'christine', 'samantha', 'debra', 'rachel', 'carolyn', 'janet', 'virginia', 'maria',
                       'heather', 'diane', 'julie', 'joyce', 'victoria', 'kelly', 'christina', 'joan', 'evelyn', 'judith',
                       'megan', 'cheryl', 'andrea', 'hannah', 'jacqueline', 'martha', 'gloria', 'sara', 'janice', 'julia',
                       'grace', 'judy', 'theresa', 'madison', 'samantha', 'lori', 'kayla', 'tiffany', 'natalie', 'denise',
                       'audrey', 'melissa', 'rozanne', 'elizabeth']
        
        if name_lower in male_names:
            return 'Male'  # Capital M to match voice system
        elif name_lower in female_names:
            return 'Female'  # Capital F to match voice system
        else:
            return 'Unknown'  # Capital U to match voice system

    def classify_gender_with_patterns(self, character_name: str, context_texts: List[str], quotes: List[str]) -> str:
        """Classify gender using hardcoded patterns and context analysis"""
        name_lower = character_name.lower().strip()
        
        # Enhanced male name patterns
        male_names = [
            'john', 'michael', 'david', 'james', 'robert', 'william', 'richard', 'charles', 'thomas', 'christopher',
            'daniel', 'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua', 'kenneth',
            'kevin', 'brian', 'george', 'timothy', 'ronald', 'jason', 'edward', 'jeffrey', 'ryan', 'jacob',
            'gary', 'nicholas', 'eric', 'jonathan', 'stephen', 'larry', 'justin', 'scott', 'brandon', 'benjamin',
            'samuel', 'frank', 'gregory', 'raymond', 'alexander', 'patrick', 'jack', 'dennis', 'jerry', 'tyler',
            'aaron', 'jose', 'henry', 'douglas', 'adam', 'peter', 'nathan', 'zachary', 'kyle', 'walter',
            'harold', 'jeremy', 'ethan', 'carl', 'keith', 'roger', 'gerald', 'arthur', 'sean', 'christian',
            'terry', 'lawrence', 'austin', 'joe', 'noah', 'jesse', 'albert', 'bryan', 'billy', 'bruce',
            'wayne', 'eugene', 'louis', 'philip', 'bobby', 'johnny', 'roy', 'ralph', 'howard', 'mister',
            'mr', 'sir', 'master', 'lord', 'duke', 'earl', 'baron', 'king', 'prince', 'emperor'
        ]
        
        # Enhanced female name patterns  
        female_names = [
            'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen',
            'nancy', 'lisa', 'betty', 'helen', 'sandra', 'donna', 'carol', 'ruth', 'sharon', 'michelle',
            'laura', 'kimberly', 'deborah', 'dorothy', 'amy', 'angela', 'brenda', 'emma', 'olivia', 'cynthia',
            'marie', 'janet', 'catherine', 'frances', 'christine', 'samantha', 'debra', 'rachel', 'carolyn',
            'virginia', 'maria', 'heather', 'diane', 'julie', 'joyce', 'victoria', 'kelly', 'christina',
            'joan', 'evelyn', 'judith', 'megan', 'cheryl', 'andrea', 'hannah', 'jacqueline', 'martha',
            'gloria', 'sara', 'janice', 'julia', 'grace', 'judy', 'theresa', 'madison', 'lori', 'kayla',
            'tiffany', 'natalie', 'denise', 'miss', 'mrs', 'ms', 'madam', 'lady', 'queen', 'princess',
            'duchess', 'countess', 'baroness', 'empress'
        ]
        
        # Check direct name match
        if name_lower in male_names:
            return 'Male'
        elif name_lower in female_names:
            return 'Female'
        
        # Check for titles and pronouns in context
        all_text = ' '.join(context_texts + quotes).lower()
        
        # Male indicators
        male_indicators = ['he ', 'him ', 'his ', 'himself', 'mr.', 'sir', 'master', 'lord', 'father', 'dad', 'brother', 'son', 'uncle', 'nephew', 'grandfather', 'grandpa']
        # Female indicators  
        female_indicators = ['she ', 'her ', 'hers ', 'herself', 'ms.', 'mrs.', 'miss', 'madam', 'lady', 'mother', 'mom', 'sister', 'daughter', 'aunt', 'niece', 'grandmother', 'grandma']
        
        male_score = sum(1 for indicator in male_indicators if indicator in all_text)
        female_score = sum(1 for indicator in female_indicators if indicator in all_text)
        
        if male_score > female_score:
            return 'Male'  # Capital M to match voice system
        elif female_score > male_score:
            return 'Female'  # Capital F to match voice system
        
        # Check name endings
        male_endings = ['son', 'sen', 'berg', 'stein', 'man', 'ton', 'ford', 'wood', 'field', 'smith', 'wright', 'miller', 'baker', 'cook', 'tailor']
        female_endings = ['son', 'sen', 'berg', 'stein', 'ette', 'ina', 'ella', 'anna', 'ella', 'ette', 'ina', 'a', 'ia', 'ina']
        
        for ending in male_endings:
            if name_lower.endswith(ending):
                return 'Male'  # Capital M to match voice system
        for ending in female_endings:
            if name_lower.endswith(ending):
                return 'Female'  # Capital F to match voice system
        
        return 'Unknown'  # Capital U to match voice system

    def classify_gender_advanced(self, name: str, context_texts: List[str], quotes: List[str]) -> str:
        """Advanced gender classification using pattern-based methods"""
        return self.classify_gender_with_patterns(name, context_texts, quotes)

    def classify_age(self, name: str, context_texts: List[str], quotes: List[str]) -> str:
        """Classify age group based on context and quotes using enhanced patterns"""
        import re
        
        all_text = ' '.join(context_texts + quotes).lower()
        
        # Enhanced age indicators with regex patterns
        child_patterns = [
            r'\b(baby|infant|toddler|child|kid|little boy|little girl)\b',
            r'\b(young|small|tiny)\b',
            r'\b(school|kindergarten|elementary)\b',
            r'\b(age\s*[0-9]|years?\s*old)\b.*[0-9]',
            r'\b(playground|toy|doll|teddy)\b'
        ]
        
        teen_patterns = [
            r'\b(teen|teenager|adolescent|youth)\b',
            r'\b(high school|college student|student)\b',
            r'\b(young adult|young man|young woman)\b',
            r'\b(age\s*1[3-9]|years?\s*old)\b.*1[3-9]',
            r'\b(prom|graduation|dormitory|campus)\b'
        ]
        
        adult_patterns = [
            r'\b(adult|grown|mature|middle-aged)\b',
            r'\b(man|woman|gentleman|lady)\b',
            r'\b(age\s*[2-9][0-9]|years?\s*old)\b.*[2-9][0-9]',
            r'\b(profession|career|job|work|office)\b',
            r'\b(marriage|married|husband|wife|spouse)\b'
        ]
        
        elderly_patterns = [
            r'\b(elderly|old|senior|aged|ancient)\b',
            r'\b(grandfather|grandmother|grandpa|grandma)\b',
            r'\b(retired|pension|pensioner)\b',
            r'\b(age\s*[6-9][0-9]|years?\s*old)\b.*[6-9][0-9]',
            r'\b(walking stick|cane|wheelchair)\b'
        ]
        
        # Count pattern matches
        child_score = sum(len(re.findall(pattern, all_text)) for pattern in child_patterns)
        teen_score = sum(len(re.findall(pattern, all_text)) for pattern in teen_patterns)
        adult_score = sum(len(re.findall(pattern, all_text)) for pattern in adult_patterns)
        elderly_score = sum(len(re.findall(pattern, all_text)) for pattern in elderly_patterns)
        
        # Additional context-based scoring
        if 'mr.' in all_text or 'sir' in all_text:
            adult_score += 2
        if 'mrs.' in all_text or 'miss' in all_text:
            adult_score += 2
        if 'dr.' in all_text or 'professor' in all_text:
            adult_score += 3
        if 'captain' in all_text or 'colonel' in all_text:
            adult_score += 2
        
        # Name-based age hints
        name_lower = name.lower()
        if any(pattern in name_lower for pattern in ['jr', 'junior', 'ii', 'iii']):
            teen_score += 3
        elif any(pattern in name_lower for pattern in ['sr', 'senior', 'old']):
            elderly_score += 3
        elif any(pattern in name_lower for pattern in ['young', 'little']):
            child_score += 2
        
        # Determine age group based on highest score
        scores = {
            'child': child_score,
            'teen': teen_score, 
            'adult': adult_score,
            'elderly': elderly_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            age_group = max(scores, key=scores.get)
            # Convert to voice-compatible age groups
            age_mapping = {
                'child': 'Young Adult',  # Map to closest available voice category
                'teen': 'Young Adult',
                'adult': 'Middle Aged',   # Map Adult to Middle Aged to match voices
                'elderly': 'Older Adult'  # Map to Older Adult to match voices
            }
            return age_mapping.get(age_group, 'Middle Aged')
        
        return 'Middle Aged'  # Default to Middle Aged to match voice system

    def analyze_personality_with_models(self, quotes: List[str], context_texts: List[str]) -> Dict[str, float]:
        """Analyze personality using enhanced rule-based patterns and regex"""
        import re
        
        all_text = ' '.join(quotes + context_texts).lower()
        
        # Initialize personality scores
        personality = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        
        # Openness patterns
        openness_patterns = [
            r'\b(creative|imaginative|artistic|innovative|original)\b',
            r'\b(curious|inquisitive|exploring|adventurous)\b',
            r'\b(open-minded|flexible|tolerant|broad-minded)\b',
            r'\b(intellectual|philosophical|abstract|complex)\b',
            r'\b(art|music|literature|poetry|culture)\b'
        ]
        
        # Conscientiousness patterns
        conscientiousness_patterns = [
            r'\b(organized|systematic|methodical|structured)\b',
            r'\b(responsible|reliable|dependable|trustworthy)\b',
            r'\b(disciplined|self-controlled|focused|determined)\b',
            r'\b(planning|preparation|careful|thorough)\b',
            r'\b(work|job|career|professional|business)\b'
        ]
        
        # Extraversion patterns
        extraversion_patterns = [
            r'\b(outgoing|sociable|talkative|gregarious)\b',
            r'\b(energetic|enthusiastic|lively|vibrant)\b',
            r'\b(social|party|friends|group|crowd)\b',
            r'\b(confident|assertive|bold|outspoken)\b',
            r'\b(leadership|leader|command|influence)\b'
        ]
        
        # Agreeableness patterns
        agreeableness_patterns = [
            r'\b(kind|gentle|caring|compassionate)\b',
            r'\b(helpful|supportive|cooperative|collaborative)\b',
            r'\b(forgiving|understanding|patient|tolerant)\b',
            r'\b(trusting|naive|innocent|pure)\b',
            r'\b(love|affection|warmth|tenderness)\b'
        ]
        
        # Neuroticism patterns
        neuroticism_patterns = [
            r'\b(anxious|worried|nervous|fearful)\b',
            r'\b(stressed|tense|overwhelmed|pressured)\b',
            r'\b(moody|emotional|sensitive|volatile)\b',
            r'\b(insecure|self-doubting|uncertain|hesitant)\b',
            r'\b(angry|irritable|frustrated|upset)\b'
        ]
        
        # Count pattern matches for each trait
        openness_score = sum(len(re.findall(pattern, all_text)) for pattern in openness_patterns)
        conscientiousness_score = sum(len(re.findall(pattern, all_text)) for pattern in conscientiousness_patterns)
        extraversion_score = sum(len(re.findall(pattern, all_text)) for pattern in extraversion_patterns)
        agreeableness_score = sum(len(re.findall(pattern, all_text)) for pattern in agreeableness_patterns)
        neuroticism_score = sum(len(re.findall(pattern, all_text)) for pattern in neuroticism_patterns)
        
        # Convert scores to personality values (0.3 to 0.8 range)
        personality['openness'] = min(0.8, max(0.3, 0.5 + openness_score * 0.05))
        personality['conscientiousness'] = min(0.8, max(0.3, 0.5 + conscientiousness_score * 0.05))
        personality['extraversion'] = min(0.8, max(0.3, 0.5 + extraversion_score * 0.05))
        personality['agreeableness'] = min(0.8, max(0.3, 0.5 + agreeableness_score * 0.05))
        personality['neuroticism'] = min(0.8, max(0.3, 0.5 + neuroticism_score * 0.05))
        
        return personality

    def generate_character_looks(self, character_name: str, gender: str, age_group: str, personality: Dict[str, float], quotes: List[str], context_texts: List[str]) -> Dict[str, str]:
        """
        Generate comprehensive character appearance descriptions based on character traits.
        """
        # Combine all text for analysis
        all_text = ' '.join([str(q.get('quote_text', '')) if isinstance(q, dict) else str(q) for q in quotes] + context_texts).lower()
        
        # Height variations by gender and age
        height_ranges = {
            'male': {
                'child': 'Short (4\'0" - 4\'6")',
                'young_adult': 'Average to tall (5\'8" - 6\'2")',
                'adult': 'Average to tall (5\'8" - 6\'2")',
                'elderly': 'Medium to tall (5\'6" - 6\'0")'
            },
            'female': {
                'child': 'Short (3\'8" - 4\'4")',
                'young_adult': 'Average (5\'4" - 5\'8")',
                'adult': 'Average (5\'4" - 5\'8")',
                'elderly': 'Medium (5\'2" - 5\'6")'
            },
            'unknown': {
                'child': 'Short (4\'0" - 4\'6")',
                'young_adult': 'Average (5\'6" - 5\'10")',
                'adult': 'Average (5\'6" - 5\'10")',
                'elderly': 'Medium (5\'4" - 5\'8")'
            }
        }
        
        # Build variations by personality traits
        build_options = {
            'high_extraversion': ['Athletic', 'Strong', 'Well-built', 'Robust'],
            'low_extraversion': ['Slender', 'Lean', 'Delicate', 'Slight'],
            'high_conscientiousness': ['Well-maintained', 'Fit', 'Toned', 'Disciplined build'],
            'high_neuroticism': ['Tense', 'Wiry', 'Nervous energy', 'Restless build'],
            'default': ['Average build', 'Medium build', 'Balanced physique', 'Proportioned']
        }
        
        # Hair colors and styles
        hair_colors = {
            'common': ['Dark brown', 'Light brown', 'Black', 'Blonde', 'Auburn'],
            'uncommon': ['Red', 'Copper', 'Chestnut', 'Ash brown', 'Golden blonde'],
            'aging': ['Gray', 'Silver', 'Gray with streaks', 'Salt and pepper', 'White']
        }
        
        hair_styles = {
            'male': {
                'young': ['Messy', 'Tousled', 'Short and neat', 'Slightly wavy', 'Styled'],
                'adult': ['Well-groomed', 'Professional cut', 'Neat', 'Classic style', 'Practical'],
                'elderly': ['Thinning', 'Receding', 'Neatly combed', 'Sparse', 'Conservative']
            },
            'female': {
                'young': ['Long and flowing', 'Wavy', 'Straight and silky', 'Curly', 'Stylishly cut'],
                'adult': ['Shoulder-length', 'Elegant updo', 'Professional style', 'Layered', 'Sophisticated'],
                'elderly': ['Short and practical', 'Neatly styled', 'Soft curls', 'Conservative cut', 'Gray and dignified']
            },
            'unknown': {
                'young': ['Neat', 'Practical', 'Styled', 'Well-kept', 'Casual'],
                'adult': ['Professional', 'Well-maintained', 'Classic', 'Practical', 'Tidy'],
                'elderly': ['Conservative', 'Neat', 'Traditional', 'Well-groomed', 'Dignified']
            }
        }
        
        # Eye colors
        eye_colors = ['Brown', 'Blue', 'Green', 'Hazel', 'Gray', 'Dark brown', 'Light blue', 'Emerald green', 'Amber']
        
        # Skin tones
        skin_tones = ['Fair', 'Light', 'Medium', 'Olive', 'Tan', 'Dark', 'Pale', 'Rosy', 'Weathered']
        
        # Clothing styles based on personality
        clothing_styles = {
            'high_openness': ['Artistic clothing', 'Creative fashion', 'Unique style', 'Bohemian attire', 'Expressive dress'],
            'high_conscientiousness': ['Professional attire', 'Well-tailored clothes', 'Neat and organized dress', 'Business formal', 'Immaculate presentation'],
            'high_extraversion': ['Bold fashion choices', 'Eye-catching attire', 'Confident style', 'Fashionable dress', 'Statement pieces'],
            'high_agreeableness': ['Comfortable clothing', 'Approachable style', 'Soft colors', 'Friendly appearance', 'Warm fashion'],
            'default': ['Practical clothing', 'Simple attire', 'Casual dress', 'Comfortable style', 'Understated fashion']
        }
        
        # Determine character traits
        gender_key = gender.lower() if gender.lower() in ['male', 'female'] else 'unknown'
        # Convert voice-compatible age groups to internal keys
        age_mapping = {
            'young adult': 'young_adult',
            'middle aged': 'adult', 
            'older adult': 'elderly'
        }
        age_key = age_mapping.get(age_group.lower(), 'adult')
        
        # Generate height
        height = height_ranges[gender_key].get(age_key, 'Average')
        
        # Generate build based on personality
        build = 'Average build'
        if personality['extraversion'] > 0.7:
            build = random.choice(build_options['high_extraversion'])
        elif personality['extraversion'] < 0.4:
            build = random.choice(build_options['low_extraversion'])
        elif personality['conscientiousness'] > 0.7:
            build = random.choice(build_options['high_conscientiousness'])
        elif personality['neuroticism'] > 0.7:
            build = random.choice(build_options['high_neuroticism'])
        else:
            build = random.choice(build_options['default'])
        
        # Generate hair
        if age_key == 'elderly':
            hair_color = random.choice(hair_colors['aging'])
        elif personality['openness'] > 0.7:
            hair_color = random.choice(hair_colors['uncommon'])
        else:
            hair_color = random.choice(hair_colors['common'])
        
        # Hair style based on gender and age
        age_style_key = 'young' if age_key in ['child', 'young_adult'] else 'adult' if age_key == 'adult' else 'elderly'
        hair_style = random.choice(hair_styles[gender_key][age_style_key])
        
        # Generate other features
        eye_color = random.choice(eye_colors)
        skin_tone = random.choice(skin_tones)
        
        # Generate clothing style based on personality
        clothing = 'Practical clothing'
        if personality['openness'] > 0.7:
            clothing = random.choice(clothing_styles['high_openness'])
        elif personality['conscientiousness'] > 0.7:
            clothing = random.choice(clothing_styles['high_conscientiousness'])
        elif personality['extraversion'] > 0.7:
            clothing = random.choice(clothing_styles['high_extraversion'])
        elif personality['agreeableness'] > 0.7:
            clothing = random.choice(clothing_styles['high_agreeableness'])
        else:
            clothing = random.choice(clothing_styles['default'])
        
        # Generate distinctive features based on personality and context
        distinctive_features = []
        if personality['openness'] > 0.7:
            distinctive_features.extend(['Creative aura', 'Expressive eyes', 'Artistic presence'])
        if personality['conscientiousness'] > 0.7:
            distinctive_features.extend(['Well-groomed appearance', 'Neat presentation', 'Organized bearing'])
        if personality['extraversion'] > 0.7:
            distinctive_features.extend(['Confident posture', 'Engaging smile', 'Charismatic presence'])
        if personality['agreeableness'] > 0.7:
            distinctive_features.extend(['Kind eyes', 'Warm expression', 'Approachable demeanor'])
        if personality['neuroticism'] > 0.7:
            distinctive_features.extend(['Intense gaze', 'Expressive features', 'Dynamic energy'])
        
        # Add some randomness and character-specific features
        general_features = ['Intelligent eyes', 'Strong jawline', 'Gentle smile', 'Piercing gaze', 'Graceful movements', 'Confident stance', 'Mysterious aura', 'Noble bearing']
        distinctive_features.extend(random.sample(general_features, min(2, len(general_features))))
        
        # Create the comprehensive physical description
        age_descriptor = {
            'child': 'young',
            'young_adult': 'young adult',
            'adult': 'adult', 
            'elderly': 'older individual'
        }.get(age_key, 'person')
        
        gender_descriptor = {
            'male': 'man',
            'female': 'woman',
            'unknown': 'individual'
        }.get(gender_key, 'person')
        
        physical_description = f"A {build.lower()} {age_descriptor} with {hair_color.lower()} hair that is typically {hair_style.lower()}. They have {eye_color.lower()} eyes and {skin_tone.lower()} skin. {character_name} is known for their {', '.join(distinctive_features[:3]).lower()} and typically dresses in {clothing.lower()}. Their overall presence suggests {random.choice(['confidence', 'mystery', 'warmth', 'intelligence', 'strength', 'grace'])} and they carry themselves with {random.choice(['poise', 'determination', 'quiet dignity', 'natural charm', 'understated elegance'])}."
        
        return {
            'physical_description': physical_description,
            'height': height,
            'build': build,
            'hair_color': hair_color,
            'hair_style': hair_style,
            'eye_color': eye_color,
            'skin_tone': skin_tone,
            'distinctive_features': ', '.join(distinctive_features[:4]),
            'typical_clothing': clothing
        }

    def save_characters_json(self, character_list: List[Dict], filename: str = "advanced_characters.json"):
        """Save advanced character data to JSON file with enhanced structure"""
        
        # Count enhanced characters
        gemini_enhanced_count = sum(1 for char in character_list if 'gemini_enhanced' in char)
        
        output_data = {
            'characters': character_list,
            'total_characters': len(character_list),
            'processing_info': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'source': 'BookNLP with advanced character processing and Gemini AI enhancement',
                'version': '4.0',
                'gemini_enhanced_characters': gemini_enhanced_count,
                'enhancement_enabled': self.gemini_enhancer.model is not None,
                'features': [
                    'Advanced character consolidation',
                    'Sophisticated quote attribution', 
                    'Multi-method personality analysis',
                    'Enhanced gender classification',
                    'Physical appearance generation',
                    'Gemini AI character enhancement',
                    'Detailed physical descriptions'
                ]
            },
            'character_schema': {
                'basic_fields': ['character_id', 'name', 'gender', 'age_group', 'mention_count'],
                'analysis_fields': ['personality', 'character_analysis', 'quotes', 'narrator_mentions'],
                'physical_fields': ['physical_description', 'height', 'build', 'hair_color', 'eye_color'],
                'enhanced_fields': ['gemini_enhanced', 'physical_description_enhanced'],
                'description': 'Enhanced characters include AI-generated insights and detailed physical descriptions'
            }
        }
        
        output_path = self.characters_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(character_list)} characters to {output_path}")
        logger.info(f"  - {gemini_enhanced_count} characters enhanced with Gemini AI")
        logger.info(f"  - {len(character_list) - gemini_enhanced_count} characters with basic analysis")
        
        return output_path

    def process_book(self, book_id: str) -> Tuple[str, List[Dict]]:
        """Main processing pipeline for advanced character extraction"""
        logger.info(f"Starting advanced processing for book: {book_id}")

        data = self.load_booknlp_output(book_id)
        if 'entities' not in data:
            raise ValueError(f"No entities data found for {book_id}")

        chapter_boundaries = []
        if 'tokens' in data:
            chapter_boundaries = self.extract_chapter_boundaries(data['tokens'])
        
        # Extract characters from entities
        person_entities = data['entities'][data['entities']['cat'] == 'PER'].copy()
        characters = {}
        
        for char_id, group in person_entities.groupby('COREF'):
            # Get the most common name for this character
            proper_names = group[group['prop'] == 'PROP']['text'].value_counts()
            if len(proper_names) > 0:
                primary_name = proper_names.index[0]
            else:
                pronouns = group[group['prop'] == 'PRON']['text'].value_counts()
                primary_name = pronouns.index[0] if len(pronouns) > 0 else f"Character_{char_id}"
            
            # Extract context
            context_texts = []
            if 'tokens' in data:
                for _, entity in group.iterrows():
                    start_token = entity['start_token']
                    end_token = entity['end_token']
                    
                    # Get surrounding context
                    context_start = max(0, start_token - 20)
                    context_end = min(len(data['tokens']), end_token + 20)
                    
                    context_tokens = data['tokens'].iloc[context_start:context_end]['word'].tolist()
                    # Filter out NaN values and convert to strings
                    context_tokens = [str(token) for token in context_tokens if pd.notna(token)]
                    context_text = ' '.join(context_tokens)
                    context_texts.append(context_text)
            
            characters[char_id] = {
                'character_id': f"char_{char_id:03d}",
                'booknlp_id': char_id,
                'name': primary_name,
                'mention_count': len(group),
                'context_texts': context_texts,
                'is_proper_name': len(proper_names) > 0,
                'all_mentions': group['text'].tolist()
            }
        
        # Consolidate characters (merge similar ones, remove pronouns)
        consolidated_characters = self.consolidate_characters(characters)
        
        # Fix quote attribution
        character_quotes = {}
        if 'quotes' in data:
            character_quotes = self.fix_quote_attribution(data['quotes'], consolidated_characters, data.get('tokens', pd.DataFrame()))
        
        # Create advanced character entries
        character_list = self.create_character_entries(consolidated_characters, character_quotes)
        
        # Save to JSON
        output_path = self.save_characters_json(character_list)
        
        logger.info(f"Completed advanced processing for {book_id}")
        return str(output_path), character_list

def find_available_books(output_dir: str = "modelsbooknlp/output") -> List[str]:
    """Find available book IDs from BookNLP output files"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    
    # Look for .entities files to identify available books
    entity_files = list(output_path.glob("*.entities"))
    book_ids = []
    
    for entity_file in entity_files:
        book_id = entity_file.stem  # Remove .entities extension
        # Check if other required files exist
        tokens_file = output_path / f"{book_id}.tokens"
        if tokens_file.exists():
            book_ids.append(book_id)
    
    return sorted(book_ids)

def main():
    """Interactive character processor with book selection"""
    print("🎭 Advanced Character Processor")
    print("=" * 50)
    
    # Use hardcoded Gemini API key
    gemini_api_key = 'AIzaSyD23o0K1sI4lghuNJhnEnUMu_B-jaUf9i4'
    print("✓ Gemini API key configured - characters will be enhanced with AI")
    print()
    
    # Find available books
    available_books = find_available_books()
    
    if not available_books:
        print("❌ No processed books found!")
        print("📁 Please ensure BookNLP has processed books in 'modelsbooknlp/output/' directory")
        print("   Required files: {book_id}.entities, {book_id}.tokens")
        return
    
    # Show available books
    print("📚 Available Books for Character Processing:")
    print("-" * 40)
    for i, book_id in enumerate(available_books, 1):
        # Check file sizes to give user info
        entities_file = Path(f"modelsbooknlp/output/{book_id}.entities")
        tokens_file = Path(f"modelsbooknlp/output/{book_id}.tokens")
        quotes_file = Path(f"modelsbooknlp/output/{book_id}.quotes")
        
        entities_size = entities_file.stat().st_size / 1024 if entities_file.exists() else 0
        tokens_size = tokens_file.stat().st_size / (1024*1024) if tokens_file.exists() else 0
        has_quotes = "✓" if quotes_file.exists() else "✗"
        
        print(f"{i}. {book_id}")
        print(f"   Entities: {entities_size:.1f}KB")
        print(f"   Tokens: {tokens_size:.1f}MB")
        print(f"   Quotes: {has_quotes}")
        print()
    
    # Let user choose
    try:
        choice = input(f"Select book to process (1-{len(available_books)}): ").strip()
        choice_num = int(choice)
        
        if not (1 <= choice_num <= len(available_books)):
            print("❌ Invalid selection")
            return
        
        selected_book_id = available_books[choice_num - 1]
        
    except ValueError:
        print("❌ Please enter a valid number")
        return
    except KeyboardInterrupt:
        print("\n👋 Processing cancelled")
        return
    
    # Confirm processing
    print(f"\n📖 Selected: {selected_book_id}")
    print("⚡ This will:")
    print("  • Extract and analyze all characters")
    print("  • Merge similar character names")
    print("  • Remove pronoun-based 'characters'")
    print("  • Analyze personality traits")
    print("  • Generate physical descriptions")
    if gemini_api_key:
        print("  • Enhance characters with Gemini AI")
    print("  • Save results to JSON file")
    print("  • Process time: 30 seconds - 2 minutes")
    print()
    
    proceed = input("🚀 Start character processing? (y/n): ").lower().strip()
    
    if proceed != 'y':
        print("👋 Processing cancelled")
        return
    
    # Initialize processor with correct paths and Gemini API key
    processor = AdvancedCharacterProcessor(
        output_dir="modelsbooknlp/output",
        characters_dir="data/processed/characters", 
        gemini_api_key=gemini_api_key
    )
    
    try:
        print(f"\n🔄 Processing characters for '{selected_book_id}'...")
        output_path, characters = processor.process_book(selected_book_id)
        
        print(f"\n✅ CHARACTER PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"📁 Characters saved to: {output_path}")
        print(f"👥 Total characters processed: {len(characters)}")
        print()
        
        # Show top characters with advanced analysis
        print(f"🎭 TOP CHARACTERS IDENTIFIED:")
        print("-" * 40)
        for i, char in enumerate(characters[:8], 1):  # Show top 8
            print(f"{i}. {char['name']} (ID: {char['character_id']})")
            print(f"   📊 Mentions: {char['mention_count']}, Importance: {char['character_analysis']['importance_score']:.2f}")
            print(f"   🧠 Personality: O={char['personality']['openness']:.2f}, C={char['personality']['conscientiousness']:.2f}, E={char['personality']['extraversion']:.2f}")
            print(f"   👤 Demographics: {char['age_group']}, {char['gender']}")
            print(f"   💬 Dialogue: {char['character_analysis']['total_dialogue_lines']}, Mentions: {char['character_analysis']['total_narrator_mentions']}")
            
            # Show Gemini enhancements if available
            if 'gemini_enhanced' in char:
                print(f"   ✨ Enhanced with Gemini AI")
            
            # Show physical description snippet
            if 'physical_description' in char and char['physical_description']:
                desc = char['physical_description'][:80] + "..." if len(char['physical_description']) > 80 else char['physical_description']
                print(f"   👁️  Description: {desc}")
            
            print()
        
        print("🎉 Character processing completed successfully!")
        print(f"📂 Next step: Run dialogue processor to create dialogue files")
        
    except Exception as e:
        logger.error(f"Advanced processing failed: {e}")
        print(f"\n❌ Processing failed: {e}")
        print("\n💡 Common issues:")
        print("  • BookNLP files might be corrupted or incomplete")
        print("  • Insufficient memory for large books")
        print("  • Network issues (if using Gemini AI)")
        raise

if __name__ == "__main__":
    main()
