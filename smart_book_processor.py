#!/usr/bin/env python3
"""
Smart Book Processor for EchoTales
Intelligently identifies real characters and generates high-quality images
"""

import json
import re
import os
import requests
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import hashlib

# Import the output checker utility
from utils.booknlp_output_checker import BookNLPOutputChecker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SmartCharacterDetector:
    """Intelligently detects real characters, not random words"""
    
    def __init__(self):
        # Words that are definitely NOT character names
        self.stop_words = {
            # Common words
            'The', 'And', 'But', 'Or', 'So', 'Then', 'Now', 'Here', 'There',
            'This', 'That', 'These', 'Those', 'When', 'Where', 'Why', 'How', 'What',
            'Who', 'Which', 'While', 'With', 'Without', 'Within', 'Would', 'Could',
            'Should', 'Will', 'Can', 'May', 'Might', 'Must', 'Shall',
            
            # Time/location words that aren't names
            'Today', 'Yesterday', 'Tomorrow', 'Morning', 'Evening', 'Night',
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
            'January', 'February', 'March', 'April', 'May', 'June', 'July',
            'August', 'September', 'October', 'November', 'December',
            
            # Directions and positions
            'North', 'South', 'East', 'West', 'Left', 'Right', 'Up', 'Down',
            'Above', 'Below', 'Before', 'After', 'Behind', 'Front', 'Back',
            
            # Generic locations (not specific character names)
            'England', 'London', 'France', 'Church', 'Cathedral', 'Castle',
            'Market', 'Street', 'Road', 'House', 'Home', 'Town', 'City',
            
            # Titles that often get picked up
            'Father', 'Mother', 'Brother', 'Sister', 'Mister', 'Mrs', 'Miss',
            'Doctor', 'Captain', 'King', 'Queen', 'Lord', 'Lady',
            
            # Common words that appear capitalized
            'All', 'Any', 'Each', 'Every', 'Some', 'Many', 'Most', 'Few',
            'First', 'Last', 'Next', 'Previous', 'Another', 'Other',
            
            # Weird words that were showing up
            'After', 'His', 'However', 'From', 'Earth', 'Its', 'Black',
            'Hey', 'Calm', 'Hehe', 'Subconsciously', 'Between', 'Chapter',
            
            # Common interjections and expressions
            'Oh', 'Ah', 'Um', 'Hmm', 'Yes', 'No', 'Well', 'Still', 'Even',
            'Just', 'Only', 'Already', 'Always', 'Never', 'Sometimes', 'Often'
        }
        
        # Common actual names (to help identify real characters)
        self.common_names = {
            'Emma', 'Thomas', 'James', 'Mary', 'John', 'Sarah', 'William', 'Elizabeth',
            'Robert', 'Margaret', 'David', 'Helen', 'Michael', 'Anna', 'Richard',
            'Catherine', 'Christopher', 'Susan', 'Matthew', 'Patricia', 'Daniel',
            'Klein', 'Zhou', 'Alice', 'Bob', 'Charlie', 'Diana', 'Frank', 'Grace'
        }
    
    def extract_real_characters(self, text: str, min_mentions: int = 5) -> List[Dict[str, Any]]:
        """Extract only real character names from text"""
        
        logger.info("Analyzing text for real character names...")
        
        # Find all capitalized words/phrases
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Count mentions
        name_counts = {}
        for name in potential_names:
            if self._is_likely_character_name(name):
                name_counts[name] = name_counts.get(name, 0) + 1
        
        # Filter by mention count and validate as real characters
        real_characters = []
        for name, count in name_counts.items():
            if count >= min_mentions and self._validate_as_character(name, text):
                character_info = {
                    'name': name,
                    'mentions': count,
                    'description': self._extract_character_description(name, text),
                    'personality': self._analyze_personality(name, text),
                    'role': self._determine_role(count, len(name_counts))
                }
                real_characters.append(character_info)
        
        # Sort by importance
        real_characters.sort(key=lambda x: x['mentions'], reverse=True)
        
        logger.info(f"Found {len(real_characters)} real characters: {[c['name'] for c in real_characters]}")
        return real_characters
    
    def _is_likely_character_name(self, name: str) -> bool:
        """Check if a name is likely to be a real character"""
        
        # Filter out obvious non-names
        if name in self.stop_words:
            return False
        
        # Must be reasonable length
        if len(name) < 3 or len(name) > 30:
            return False
        
        # Check if it looks like a name
        words = name.split()
        
        # Single word names
        if len(words) == 1:
            # Must look like a proper name (not be a common word)
            if name.lower() in ['it', 'he', 'she', 'his', 'her', 'him', 'them', 'they']:
                return False
            if name in self.common_names:
                return True
            # Check if it has name-like qualities
            return self._looks_like_name(name)
        
        # Multi-word names (like "John Smith")
        elif len(words) == 2:
            # Both words should look like name parts
            return all(self._looks_like_name(word) for word in words)
        
        # Three or more words - be more restrictive
        elif len(words) >= 3:
            # Only if it's a very common pattern like "Mary Jane Smith"
            return all(self._looks_like_name(word) for word in words) and len(words) <= 3
        
        return False
    
    def _looks_like_name(self, word: str) -> bool:
        """Check if a single word looks like a name"""
        
        # Common name patterns
        if word in self.common_names:
            return True
        
        # Names typically end with certain sounds/letters
        name_endings = ['a', 'e', 'i', 'o', 'y', 'n', 's', 't', 'r', 'h', 'k', 'm']
        if word.lower().endswith(tuple(name_endings)) and len(word) >= 3:
            # Additional check: not common words
            common_words = {'the', 'and', 'but', 'for', 'you', 'all', 'any', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
            return word.lower() not in common_words
        
        return False
    
    def _validate_as_character(self, name: str, text: str) -> bool:
        """Validate that this is actually used as a character in the text"""
        
        try:
            # Look for character-like usage patterns
            character_patterns = [
                rf'\b{re.escape(name)}\s+said\b',
                rf'\b{re.escape(name)}\s+walked\b',
                rf'\b{re.escape(name)}\s+looked\b',
                rf'\b{re.escape(name)}\s+smiled\b',
                rf'\b{re.escape(name)}\s+went\b',
                rf'\b{re.escape(name)}\s+was\b',
                rf'"[^"]*"\s+{re.escape(name)}\b',  # Dialogue attribution
                rf'\b{re.escape(name)}\'s\b',  # Possessive
            ]
            
            character_usage_count = 0
            for pattern in character_patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    character_usage_count += len(matches)
                except re.error:
                    # Skip problematic patterns
                    continue
            
            # Must be used in character-like ways at least 2 times
            return character_usage_count >= 2
        except Exception:
            # If any error occurs, just return False
            return False
    
    def _extract_character_description(self, name: str, text: str) -> str:
        """Extract a meaningful description of the character"""
        
        try:
            # Find sentences that mention the character
            sentences = re.split(r'[.!?]+', text)
            relevant_sentences = []
            
            for sentence in sentences:
                if name in sentence and len(sentence.strip()) > 20:
                    # Clean up the sentence
                    clean_sentence = sentence.strip()
                    if clean_sentence and not clean_sentence.startswith('"'):
                        relevant_sentences.append(clean_sentence)
            
            # Return the most descriptive sentences
            if relevant_sentences:
                # Prefer longer, more descriptive sentences
                relevant_sentences.sort(key=len, reverse=True)
                return '. '.join(relevant_sentences[:2][:300]) + '.'
            else:
                return f"A character named {name} who appears in the story."
        except Exception:
            return f"A character named {name} who appears in the story."
    
    def _analyze_personality(self, name: str, text: str) -> List[str]:
        """Analyze character personality from context"""
        
        # Get text around character mentions
        context = self._get_character_context(name, text, 150)
        context_lower = context.lower()
        
        personality_traits = []
        
        trait_indicators = {
            'kind': ['kind', 'gentle', 'caring', 'helpful', 'good', 'nice'],
            'brave': ['brave', 'courageous', 'bold', 'fearless', 'daring'],
            'intelligent': ['smart', 'clever', 'wise', 'intelligent', 'brilliant'],
            'cheerful': ['smiled', 'laughed', 'happy', 'cheerful', 'bright'],
            'serious': ['serious', 'stern', 'grave', 'solemn', 'focused'],
            'mysterious': ['mysterious', 'secretive', 'enigmatic', 'hidden']
        }
        
        for trait, indicators in trait_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                personality_traits.append(trait)
        
        return personality_traits[:3] if personality_traits else ['unknown']
    
    def _get_character_context(self, name: str, text: str, context_size: int) -> str:
        """Get text context around character mentions"""
        
        contexts = []
        words = text.split()
        
        for i, word in enumerate(words):
            if name in word:
                start = max(0, i - context_size // 2)
                end = min(len(words), i + context_size // 2)
                context = ' '.join(words[start:end])
                contexts.append(context)
                
                if len(contexts) >= 3:  # Limit context gathering
                    break
        
        return ' '.join(contexts)
    
    def _determine_role(self, mentions: int, total_characters: int) -> str:
        """Determine character's role based on mentions"""
        
        if mentions >= 20:
            return "protagonist"
        elif mentions >= 10:
            return "major"
        elif mentions >= 5:
            return "supporting"
        else:
            return "minor"


class HighQualityImageGenerator:
    """Generate high-quality images with proper prompts"""
    
    def __init__(self):
        self.output_dir = Path("modelsbooknlp/generated")
        self.characters_dir = self.output_dir / "character_portraits"
        self.scenes_dir = self.output_dir / "scene_images"
        
        # Create directories
        for dir_path in [self.characters_dir, self.scenes_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_character_portrait(self, character: Dict[str, Any]) -> Optional[str]:
        """Generate a high-quality character portrait"""
        
        name = character['name']
        description = character['description']
        personality = character['personality']
        
        logger.info(f"Generating portrait for character: {name}")
        
        # Create detailed character prompt
        prompt = self._build_character_prompt(name, description, personality)
        
        # Generate image
        filename = f"{name.lower().replace(' ', '_')}_portrait"
        image_path = self._generate_ai_image(prompt, filename, self.characters_dir)
        
        if image_path:
            logger.info(f"✅ Generated character portrait: {name}")
            return image_path
        else:
            logger.warning(f"❌ Failed to generate portrait for: {name}")
            return None
    
    def generate_scene_image(self, scene_description: str, characters: List[str], 
                           setting: str, mood: str) -> Optional[str]:
        """Generate a high-quality scene image"""
        
        logger.info(f"Generating scene image: {scene_description[:50]}...")
        
        # Create scene prompt
        prompt = self._build_scene_prompt(scene_description, characters, setting, mood)
        
        # Generate filename
        scene_hash = hashlib.md5(scene_description.encode()).hexdigest()[:8]
        filename = f"scene_{scene_hash}"
        
        image_path = self._generate_ai_image(prompt, filename, self.scenes_dir)
        
        if image_path:
            logger.info(f"✅ Generated scene image")
            return image_path
        else:
            logger.warning(f"❌ Failed to generate scene image")
            return None
    
    def _build_character_prompt(self, name: str, description: str, personality: List[str]) -> str:
        """Build detailed character prompt"""
        
        prompt_parts = []
        
        # Base prompt
        prompt_parts.append("professional character portrait")
        
        # Extract appearance from description
        appearance = self._extract_appearance(description)
        if appearance:
            prompt_parts.append(appearance)
        else:
            prompt_parts.append("adult person")
        
        # Personality-based appearance
        if personality and personality[0] != 'unknown':
            personality_look = self._personality_to_appearance(personality)
            if personality_look:
                prompt_parts.append(personality_look)
        
        # Style and quality
        prompt_parts.extend([
            "realistic digital art",
            "detailed facial features",
            "professional lighting",
            "high quality",
            "portrait photography style",
            "clean background"
        ])
        
        prompt = ", ".join(prompt_parts)
        return self._clean_prompt(prompt)
    
    def _build_scene_prompt(self, description: str, characters: List[str], 
                          setting: str, mood: str) -> str:
        """Build detailed scene prompt"""
        
        prompt_parts = []
        
        # Setting
        if setting and setting != "unknown location":
            prompt_parts.append(f"{setting}")
        
        # Character count
        if characters:
            if len(characters) == 1:
                prompt_parts.append("single person")
            elif len(characters) == 2:
                prompt_parts.append("two people")
            else:
                prompt_parts.append("group of people")
        
        # Mood and atmosphere
        mood_descriptions = {
            "cheerful": "bright, cheerful atmosphere, warm lighting",
            "mysterious": "mysterious, atmospheric lighting",
            "tense": "dramatic, tense atmosphere",
            "peaceful": "calm, peaceful environment",
            "romantic": "romantic, soft lighting"
        }
        
        if mood in mood_descriptions:
            prompt_parts.append(mood_descriptions[mood])
        
        # Style
        prompt_parts.extend([
            "cinematic composition",
            "detailed illustration",
            "professional digital art",
            "high quality"
        ])
        
        prompt = ", ".join(prompt_parts)
        return self._clean_prompt(prompt)
    
    def _extract_appearance(self, description: str) -> str:
        """Extract appearance details from description"""
        
        appearance_parts = []
        desc_lower = description.lower()
        
        # Age
        if 'young' in desc_lower:
            appearance_parts.append('young adult')
        elif 'old' in desc_lower:
            appearance_parts.append('elderly person')
        else:
            appearance_parts.append('adult')
        
        # Hair
        hair_colors = ['blonde', 'brown', 'black', 'red', 'gray', 'white']
        for color in hair_colors:
            if color in desc_lower:
                appearance_parts.append(f'{color} hair')
                break
        
        # Eyes
        if 'blue eyes' in desc_lower:
            appearance_parts.append('blue eyes')
        elif 'green eyes' in desc_lower:
            appearance_parts.append('green eyes')
        elif 'brown eyes' in desc_lower:
            appearance_parts.append('brown eyes')
        
        return ', '.join(appearance_parts) if appearance_parts else 'person'
    
    def _personality_to_appearance(self, personality: List[str]) -> str:
        """Convert personality to visual cues"""
        
        personality_looks = {
            'kind': 'warm, friendly expression',
            'cheerful': 'bright, smiling expression',
            'serious': 'focused, serious expression',
            'mysterious': 'enigmatic, thoughtful expression',
            'brave': 'confident, determined look'
        }
        
        looks = []
        for trait in personality[:2]:  # Max 2 traits
            if trait in personality_looks:
                looks.append(personality_looks[trait])
        
        return ', '.join(looks)
    
    def _generate_ai_image(self, prompt: str, filename: str, output_dir: Path) -> Optional[str]:
        """Generate image using AI API"""
        
        try:
            # Use Pollinations AI (free and good quality)
            clean_prompt = prompt.replace(" ", "%20").replace(",", "%2C")
            url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=512&height=768&nologo=true&enhance=true"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 10000:  # Minimum quality check
                image_path = output_dir / f"{filename}.jpg"
                
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                if image_path.exists() and image_path.stat().st_size > 10000:
                    return str(image_path)
            
        except Exception as e:
            logger.warning(f"AI image generation failed: {e}")
        
        return None
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and optimize prompt"""
        
        # Remove special characters that might cause issues
        prompt = re.sub(r'[^\w\s,.-]', '', prompt)
        
        # Limit length
        if len(prompt) > 300:
            prompt = prompt[:300]
        
        return prompt.strip()


class SmartBookProcessor:
    """Main processor that generates quality content efficiently"""
    
    def __init__(self):
        self.character_detector = SmartCharacterDetector()
        self.image_generator = HighQualityImageGenerator()
        self.output_dir = Path("modelsbooknlp/generated")
        self.json_dir = self.output_dir / "profiles"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.output_checker = BookNLPOutputChecker()
    
    def process_book(self, book_text: str, title: str, author: str, skip_if_exists: bool = True) -> Dict[str, Any]:
        """Process book with smart character detection and quality image generation"""
        
        logger.info(f"🎬 Smart processing: '{title}' by {author}")
        logger.info("📋 Only generating content for REAL characters and key scenes")
        
        # Generate consistent book_id
        book_id = self.output_checker.generate_book_id(book_text, title)
        
        # Check if processing already exists and should be skipped
        if skip_if_exists and self.output_checker.check_smart_processor_output_exists(book_id):
            logger.info(f"⏩ Skipping SmartBookProcessor - output already exists for: {title}")
            return self._load_existing_results(title, author, book_text)
        
        try:
            # Extract REAL characters only (not random words)
            real_characters = self.character_detector.extract_real_characters(book_text, min_mentions=8)
            
            # Limit to main characters to avoid spam
            main_characters = real_characters[:6]  # Max 6 main characters
            
            if not main_characters:
                logger.warning("No real characters found! Check the text quality.")
                return {"success": False, "error": "No real characters detected"}
            
            logger.info(f"🎭 Processing {len(main_characters)} main characters")
            
            # Generate character portraits
            generated_portraits = []
            for character in main_characters:
                portrait_path = self.image_generator.generate_character_portrait(character)
                
                if portrait_path:
                    character['portrait_path'] = portrait_path
                    generated_portraits.append(character)
                
                # Save character profile
                char_file = self._save_character_profile(character)
                
                # Respectful delay for API
                time.sleep(3)
            
            # Find 3-5 key scenes (not every paragraph)
            key_scenes = self._find_key_scenes(book_text, main_characters, max_scenes=4)
            
            # Generate scene images
            generated_scenes = []
            for i, scene in enumerate(key_scenes, 1):
                scene_image = self.image_generator.generate_scene_image(
                    scene['description'],
                    scene['characters'],
                    scene['setting'],
                    scene['mood']
                )
                
                if scene_image:
                    scene['image_path'] = scene_image
                    generated_scenes.append(scene)
                
                # Save scene profile
                scene_file = self._save_scene_profile(scene, i)
                
                time.sleep(3)
            
            # Create summary
            result = {
                "success": True,
                "book_info": {
                    "title": title,
                    "author": author,
                    "word_count": len(book_text.split()),
                    "processed_at": datetime.now().isoformat()
                },
                "characters_processed": len(main_characters),
                "portraits_generated": len(generated_portraits),
                "scenes_processed": len(key_scenes),
                "scene_images_generated": len(generated_scenes),
                "total_files": len(main_characters) + len(key_scenes),
                "character_names": [c['name'] for c in main_characters]
            }
            
            # Save master summary
            self._save_master_summary(result)
            
            logger.info("✅ Smart processing complete!")
            logger.info(f"📊 Generated content for {len(main_characters)} characters and {len(key_scenes)} scenes")
            logger.info(f"👥 Characters: {', '.join(result['character_names'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"Smart processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _find_key_scenes(self, text: str, characters: List[Dict], max_scenes: int) -> List[Dict]:
        """Find key dramatic scenes"""
        
        # Split into larger chunks (not tiny segments)
        chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 150]
        
        scored_scenes = []
        
        for chunk in chunks:
            score = self._score_scene_importance(chunk, characters)
            
            if score >= 5:  # Only high-importance scenes
                scene = {
                    'description': chunk[:200],  # First 200 chars
                    'characters': self._find_characters_in_text(chunk, characters),
                    'setting': self._detect_setting(chunk),
                    'mood': self._detect_mood(chunk),
                    'importance_score': score
                }
                scored_scenes.append(scene)
        
        # Sort by importance and take top scenes
        scored_scenes.sort(key=lambda x: x['importance_score'], reverse=True)
        return scored_scenes[:max_scenes]
    
    def _score_scene_importance(self, text: str, characters: List[Dict]) -> int:
        """Score how important/dramatic a scene is"""
        
        score = 0
        text_lower = text.lower()
        
        # Dialogue = more important
        if text.count('"') >= 2:
            score += 3
        
        # Multiple main characters = important
        char_count = 0
        for char in characters:
            if char['name'].lower() in text_lower:
                char_count += 1
        score += char_count * 2
        
        # Action/emotion words
        important_words = ['said', 'exclaimed', 'whispered', 'shouted', 'realized', 'discovered', 'found', 'saw', 'heard']
        score += sum(1 for word in important_words if word in text_lower)
        
        # Emotional content
        emotion_words = ['surprised', 'shocked', 'amazed', 'worried', 'frightened', 'delighted', 'angry']
        score += sum(1 for word in emotion_words if word in text_lower) * 2
        
        return score
    
    def _find_characters_in_text(self, text: str, characters: List[Dict]) -> List[str]:
        """Find which characters are in this text"""
        
        found = []
        text_lower = text.lower()
        
        for char in characters:
            if char['name'].lower() in text_lower:
                found.append(char['name'])
        
        return found
    
    def _detect_setting(self, text: str) -> str:
        """Detect scene setting"""
        
        text_lower = text.lower()
        
        settings = {
            'marketplace': ['market', 'stall', 'vendor', 'crowd'],
            'cathedral': ['cathedral', 'church', 'altar', 'priest'],
            'street': ['street', 'road', 'path', 'walked'],
            'home': ['home', 'house', 'room', 'door'],
            'forest': ['forest', 'trees', 'woods'],
            'castle': ['castle', 'tower', 'hall']
        }
        
        for setting, keywords in settings.items():
            if sum(1 for kw in keywords if kw in text_lower) >= 2:
                return setting
        
        return 'unknown location'
    
    def _detect_mood(self, text: str) -> str:
        """Detect scene mood"""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['smiled', 'laughed', 'happy', 'delighted']):
            return 'cheerful'
        elif any(word in text_lower for word in ['worried', 'frightened', 'anxious', 'tense']):
            return 'tense'
        elif any(word in text_lower for word in ['mysterious', 'strange', 'secret', 'hidden']):
            return 'mysterious'
        elif any(word in text_lower for word in ['peaceful', 'calm', 'gentle', 'quiet']):
            return 'peaceful'
        else:
            return 'neutral'
    
    def _save_character_profile(self, character: Dict[str, Any]) -> str:
        """Save character profile"""
        
        filename = f"{character['name'].lower().replace(' ', '_')}_profile.json"
        filepath = self.json_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(character, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _save_scene_profile(self, scene: Dict[str, Any], scene_num: int) -> str:
        """Save scene profile"""
        
        filename = f"key_scene_{scene_num:02d}.json"
        filepath = self.json_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scene, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _load_existing_results(self, title: str, author: str, book_text: str) -> Dict[str, Any]:
        """Load existing processing results"""
        
        try:
            # Load summary file
            summary_file = self.json_dir / "book_processing_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                
                # Update with current request info
                existing_result["book_info"]["title"] = title
                existing_result["book_info"]["author"] = author
                existing_result["book_info"]["word_count"] = len(book_text.split())
                existing_result["skipped_processing"] = True
                existing_result["message"] = "Using existing SmartBookProcessor results"
                
                return existing_result
        
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
        
        # Fallback - create basic result from existing files
        profiles_dir = self.json_dir
        portraits_dir = self.output_dir / "character_portraits"
        scenes_dir = self.output_dir / "scene_images"
        
        character_files = list(profiles_dir.glob("*_profile.json")) if profiles_dir.exists() else []
        portrait_files = list(portraits_dir.glob("*.jpg")) if portraits_dir.exists() else []
        scene_files = list(scenes_dir.glob("*.jpg")) if scenes_dir.exists() else []
        
        character_names = []
        for char_file in character_files:
            try:
                with open(char_file, 'r', encoding='utf-8') as f:
                    char_data = json.load(f)
                    if 'name' in char_data:
                        character_names.append(char_data['name'])
            except:
                pass
        
        return {
            "success": True,
            "book_info": {
                "title": title,
                "author": author,
                "word_count": len(book_text.split()),
                "processed_at": datetime.now().isoformat()
            },
            "characters_processed": len(character_files),
            "portraits_generated": len(portrait_files),
            "scenes_processed": len(scene_files),
            "scene_images_generated": len(scene_files),
            "total_files": len(character_files) + len(scene_files),
            "character_names": character_names,
            "skipped_processing": True,
            "message": "Using existing SmartBookProcessor results"
        }
    
    def _save_master_summary(self, result: Dict[str, Any]) -> str:
        """Save master summary"""
        
        filepath = self.json_dir / "book_processing_summary.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return str(filepath)


def main():
    """Test the smart processor"""
    
    processor = SmartBookProcessor()
    
    # Test with sample
    sample_text = '''Emma walked through the busy marketplace in London, her basket clutched tightly in her hand. The morning sun cast long shadows between the wooden stalls.

"Fresh apples! Best in all of England!" called out Thomas, the fruit vendor, waving to catch her attention.

Emma smiled and approached his stall. "Good morning, Thomas. How much for a dozen?"

"For you, my dear Emma, just two shillings," Thomas replied with a grin. "Your father was always good to me."

James appeared suddenly from behind a cart. "Emma! I've been looking for you everywhere. You won't believe what I discovered at the cathedral!"

"What is it, James?" Emma asked, concerned by her brother's excited expression.

"Father McKenzie showed me an old letter - it's about our grandfather's service in the war. There are secrets our family never knew!"

Emma gasped, dropping her basket. The apples scattered across the cobblestones as she processed this shocking revelation.'''
    
    result = processor.process_book(sample_text, "The Marketplace Mystery", "Test Author")
    
    if result["success"]:
        print("✅ Smart processing complete!")
        print(f"👥 Real characters found: {result['character_names']}")
        print(f"🎨 Portraits generated: {result['portraits_generated']}")
        print(f"🎬 Scene images: {result['scene_images_generated']}")
        print(f"📁 Total files: {result['total_files']}")
    else:
        print(f"❌ Processing failed: {result['error']}")


if __name__ == "__main__":
    main()