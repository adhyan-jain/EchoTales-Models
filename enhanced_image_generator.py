#!/usr/bin/env python3
"""
Enhanced Image Generator for EchoTales
Creates proper character portraits and scene images using AI image generation APIs
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
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class EnhancedImageGenerator:
    """Enhanced image generator with proper AI image generation"""
    
    def __init__(self):
        self.output_dir = Path("data/generated")
        self.characters_dir = self.output_dir / "characters"
        self.scenes_dir = self.output_dir / "scenes"
        self.backgrounds_dir = self.output_dir / "backgrounds"
        
        # Create directories
        for dir_path in [self.characters_dir, self.scenes_dir, self.backgrounds_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # AI Image Generation APIs (free alternatives)
        self.image_apis = {
            "pollinations": {
                "url": "https://image.pollinations.ai/prompt/",
                "method": "GET",
                "params": {"width": 512, "height": 768}  # Portrait format
            },
            "picsum": {
                "url": "https://picsum.photos/",
                "method": "GET",
                "fallback": True
            }
        }
        
        logger.info("Enhanced Image Generator initialized")
    
    def generate_character_portrait(self, character_name: str, description: str, 
                                  personality: List[str], age: str = "adult") -> Optional[str]:
        """Generate a proper character portrait"""
        
        logger.info(f"Generating character portrait for {character_name}")
        
        # Create detailed character prompt
        prompt = self._create_character_prompt(character_name, description, personality, age)
        
        # Try to generate image
        image_path = self._generate_image(
            prompt=prompt,
            filename=f"{character_name.lower().replace(' ', '_')}_portrait",
            output_dir=self.characters_dir,
            size=(512, 768)  # Portrait format
        )
        
        if image_path:
            logger.info(f"✅ Generated character portrait: {Path(image_path).name}")
            return image_path
        
        logger.warning(f"Failed to generate portrait for {character_name}")
        return None
    
    def generate_scene_image(self, scene_description: str, characters: List[str], 
                           setting: str, mood: str, chapter_title: str = "") -> Optional[str]:
        """Generate a proper scene image"""
        
        logger.info(f"Generating scene image: {scene_description[:50]}...")
        
        # Create detailed scene prompt
        prompt = self._create_scene_prompt(scene_description, characters, setting, mood)
        
        # Generate filename
        scene_hash = hashlib.md5(scene_description.encode()).hexdigest()[:8]
        filename = f"scene_{scene_hash}"
        
        # Try to generate image
        image_path = self._generate_image(
            prompt=prompt,
            filename=filename,
            output_dir=self.scenes_dir,
            size=(768, 512)  # Landscape format for scenes
        )
        
        if image_path:
            logger.info(f"✅ Generated scene image: {Path(image_path).name}")
            return image_path
        
        logger.warning("Failed to generate scene image")
        return None
    
    def generate_background_image(self, setting: str, time_of_day: str = "day", 
                                weather: str = "clear") -> Optional[str]:
        """Generate a background/setting image"""
        
        logger.info(f"Generating background for {setting}")
        
        # Create background prompt
        prompt = self._create_background_prompt(setting, time_of_day, weather)
        
        # Generate filename
        setting_clean = re.sub(r'[^\w\s-]', '', setting).replace(' ', '_')
        filename = f"bg_{setting_clean}"
        
        # Try to generate image
        image_path = self._generate_image(
            prompt=prompt,
            filename=filename,
            output_dir=self.backgrounds_dir,
            size=(1024, 768)  # Wide format for backgrounds
        )
        
        if image_path:
            logger.info(f"✅ Generated background: {Path(image_path).name}")
            return image_path
        
        logger.warning(f"Failed to generate background for {setting}")
        return None
    
    def _create_character_prompt(self, name: str, description: str, 
                               personality: List[str], age: str) -> str:
        """Create detailed prompt for character portrait"""
        
        # Base character prompt
        prompt_parts = []
        
        # Character description
        if description and len(description) > 20:
            # Extract appearance details
            appearance = self._extract_appearance_details(description)
            prompt_parts.append(f"portrait of {appearance}")
        else:
            prompt_parts.append(f"portrait of a person")
        
        # Age context
        age_descriptors = {
            "young": "young, youthful features",
            "adult": "mature, adult features",
            "elderly": "older, wise appearance"
        }
        prompt_parts.append(age_descriptors.get(age, "adult features"))
        
        # Personality-based appearance
        if personality:
            personality_desc = self._personality_to_appearance(personality)
            if personality_desc:
                prompt_parts.append(personality_desc)
        
        # Style and quality
        prompt_parts.extend([
            "realistic portrait",
            "detailed facial features",
            "professional lighting",
            "high quality",
            "photorealistic",
            "4k resolution"
        ])
        
        prompt = ", ".join(prompt_parts)
        
        # Clean and limit prompt
        prompt = re.sub(r'[^\w\s,.-]', '', prompt)
        prompt = prompt[:400]  # Reasonable length
        
        logger.info(f"Character prompt: {prompt[:100]}...")
        return prompt
    
    def _create_scene_prompt(self, description: str, characters: List[str], 
                           setting: str, mood: str) -> str:
        """Create detailed prompt for scene image"""
        
        prompt_parts = []
        
        # Main scene setting
        if setting and setting != "unspecified location":
            prompt_parts.append(f"{setting} scene")
        
        # Character count and interaction
        if characters:
            if len(characters) == 1:
                prompt_parts.append("single character")
            elif len(characters) == 2:
                prompt_parts.append("two characters interacting")
            else:
                prompt_parts.append("multiple characters in scene")
        
        # Extract action from description
        action = self._extract_scene_action(description)
        if action:
            prompt_parts.append(action)
        
        # Mood and atmosphere
        mood_descriptors = {
            "cheerful": "bright, cheerful atmosphere, warm lighting",
            "tense": "dramatic tension, moody lighting",
            "mysterious": "mysterious atmosphere, shadow and light",
            "peaceful": "peaceful, serene environment",
            "romantic": "romantic ambiance, soft lighting",
            "energetic": "dynamic, energetic scene"
        }
        
        if mood in mood_descriptors:
            prompt_parts.append(mood_descriptors[mood])
        
        # Style and quality
        prompt_parts.extend([
            "cinematic composition",
            "detailed illustration",
            "professional artwork",
            "high quality",
            "digital art"
        ])
        
        prompt = ", ".join(prompt_parts)
        prompt = re.sub(r'[^\w\s,.-]', '', prompt)
        prompt = prompt[:400]
        
        logger.info(f"Scene prompt: {prompt[:100]}...")
        return prompt
    
    def _create_background_prompt(self, setting: str, time_of_day: str, weather: str) -> str:
        """Create prompt for background image"""
        
        prompt_parts = []
        
        # Main setting
        setting_prompts = {
            "marketplace": "medieval marketplace, market stalls, cobblestone streets",
            "cathedral": "grand cathedral interior, stained glass windows, stone architecture",
            "London": "Victorian London street, historic buildings, cobblestone roads",
            "village": "quaint village scene, rural cottages, countryside",
            "castle": "medieval castle, stone walls, towers",
            "forest": "enchanted forest, tall trees, natural lighting",
            "church": "church interior, wooden pews, altar"
        }
        
        setting_desc = setting_prompts.get(setting.lower(), f"{setting} landscape")
        prompt_parts.append(setting_desc)
        
        # Time of day
        time_lighting = {
            "morning": "morning light, golden hour",
            "day": "natural daylight, clear sky",
            "evening": "evening light, sunset glow",
            "night": "night scene, moonlight, atmospheric"
        }
        
        if time_of_day in time_lighting:
            prompt_parts.append(time_lighting[time_of_day])
        
        # Weather
        weather_effects = {
            "sunny": "bright sunny day, clear skies",
            "cloudy": "overcast sky, soft diffused light",
            "rainy": "rainy atmosphere, wet surfaces",
            "foggy": "misty fog, atmospheric haze"
        }
        
        if weather in weather_effects:
            prompt_parts.append(weather_effects[weather])
        
        # Style
        prompt_parts.extend([
            "detailed environment art",
            "atmospheric perspective",
            "professional digital art",
            "concept art style",
            "high resolution"
        ])
        
        prompt = ", ".join(prompt_parts)
        prompt = re.sub(r'[^\w\s,.-]', '', prompt)
        prompt = prompt[:400]
        
        logger.info(f"Background prompt: {prompt[:100]}...")
        return prompt
    
    def _generate_image(self, prompt: str, filename: str, output_dir: Path, 
                      size: tuple = (512, 512)) -> Optional[str]:
        """Generate image using AI APIs"""
        
        # Try Pollinations AI first (best quality)
        image_path = self._try_pollinations_api(prompt, filename, output_dir, size)
        if image_path:
            return image_path
        
        # Fallback to other methods if needed
        logger.warning("Primary AI image generation failed, using fallback")
        return None
    
    def _try_pollinations_api(self, prompt: str, filename: str, output_dir: Path, 
                            size: tuple) -> Optional[str]:
        """Try Pollinations AI image generation"""
        
        try:
            # Clean prompt for URL
            clean_prompt = prompt.replace(" ", "%20").replace(",", "%2C").replace("&", "%26")
            
            # Construct URL with size parameters
            url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width={size[0]}&height={size[1]}&nologo=true&enhance=true"
            
            # Make request with timeout
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200 and len(response.content) > 5000:  # Minimum size check
                # Save image
                image_path = output_dir / f"{filename}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify image was saved properly
                if image_path.exists() and image_path.stat().st_size > 5000:
                    return str(image_path)
                else:
                    logger.warning(f"Generated image too small: {image_path.stat().st_size if image_path.exists() else 0} bytes")
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Pollinations API request failed: {e}")
        except Exception as e:
            logger.warning(f"Image generation error: {e}")
        
        return None
    
    def _extract_appearance_details(self, description: str) -> str:
        """Extract appearance details from character description"""
        
        # Look for appearance keywords
        appearance_words = []
        desc_lower = description.lower()
        
        # Physical features
        if 'tall' in desc_lower:
            appearance_words.append('tall')
        elif 'short' in desc_lower:
            appearance_words.append('short')
        
        # Hair color
        hair_colors = ['blonde', 'brown', 'black', 'red', 'gray', 'white']
        for color in hair_colors:
            if color in desc_lower:
                appearance_words.append(f'{color} hair')
                break
        
        # Eye color
        if 'blue eyes' in desc_lower:
            appearance_words.append('blue eyes')
        elif 'green eyes' in desc_lower:
            appearance_words.append('green eyes')
        elif 'brown eyes' in desc_lower:
            appearance_words.append('brown eyes')
        
        # Age descriptors
        if 'young' in desc_lower:
            appearance_words.append('youthful')
        elif 'old' in desc_lower:
            appearance_words.append('elderly')
        
        if appearance_words:
            return ', '.join(appearance_words) + ' person'
        else:
            return 'person'
    
    def _personality_to_appearance(self, personality: List[str]) -> str:
        """Convert personality traits to visual appearance cues"""
        
        appearance_cues = []
        
        personality_mapping = {
            'kind': 'warm, friendly expression',
            'cheerful': 'bright, smiling expression',
            'serious': 'stern, focused expression',
            'mysterious': 'enigmatic, thoughtful expression',
            'brave': 'confident, determined look',
            'intelligent': 'wise, thoughtful appearance'
        }
        
        for trait in personality:
            if trait in personality_mapping:
                appearance_cues.append(personality_mapping[trait])
        
        return ', '.join(appearance_cues[:2])  # Limit to 2 cues
    
    def _extract_scene_action(self, description: str) -> str:
        """Extract main action from scene description"""
        
        desc_lower = description.lower()
        
        # Common actions
        if 'walking' in desc_lower or 'walked' in desc_lower:
            return 'characters walking'
        elif 'talking' in desc_lower or 'conversation' in desc_lower:
            return 'characters in conversation'
        elif 'looking' in desc_lower or 'watching' in desc_lower:
            return 'characters observing'
        elif 'sitting' in desc_lower:
            return 'characters sitting'
        elif 'standing' in desc_lower:
            return 'characters standing'
        else:
            return 'character interaction'


class SmartBookProcessor:
    """Improved book processor that generates fewer, higher-quality files"""
    
    def __init__(self):
        self.image_generator = EnhancedImageGenerator()
        self.output_dir = Path("data/generated")
        self.json_dir = self.output_dir / "json"
        self.json_dir.mkdir(parents=True, exist_ok=True)
    
    def process_book_smart(self, book_text: str, title: str, author: str, 
                          max_characters: int = 8, max_scenes: int = 12) -> Dict[str, Any]:
        """Process book with intelligent filtering - generate only key content"""
        
        logger.info(f"Smart processing: {title} by {author}")
        logger.info(f"Limiting to {max_characters} main characters and {max_scenes} key scenes")
        
        try:
            # Analyze book structure
            analysis = self._analyze_book_structure(book_text, title, author)
            
            # Extract only main characters (not every proper noun)
            main_characters = self._extract_main_characters(book_text, max_characters)
            
            # Find key scenes (dramatic moments, not every paragraph)
            key_scenes = self._identify_key_scenes(book_text, main_characters, max_scenes)
            
            # Generate character portraits for main characters only
            character_files = []
            for character in main_characters:
                portrait_path = self.image_generator.generate_character_portrait(
                    character['name'],
                    character['description'],
                    character['personality'],
                    character['age']
                )
                
                if portrait_path:
                    character['portrait_path'] = portrait_path
                    char_file = self._save_character_profile(character)
                    character_files.append(char_file)
                
                # Small delay to be respectful to API
                time.sleep(2)
            
            # Generate scene images for key scenes only
            scene_files = []
            for i, scene in enumerate(key_scenes, 1):
                scene_image = self.image_generator.generate_scene_image(
                    scene['description'],
                    scene['characters'],
                    scene['setting'],
                    scene['mood'],
                    f"Scene {i}"
                )
                
                if scene_image:
                    scene['image_path'] = scene_image
                    scene_file = self._save_scene_profile(scene, i)
                    scene_files.append(scene_file)
                
                # Small delay
                time.sleep(3)
            
            # Generate a few background images for main settings
            settings = self._extract_main_settings(book_text)
            background_files = []
            for setting in settings[:4]:  # Max 4 backgrounds
                bg_image = self.image_generator.generate_background_image(setting)
                if bg_image:
                    background_files.append(bg_image)
                time.sleep(2)
            
            # Create comprehensive summary
            result = {
                "success": True,
                "book_info": analysis,
                "main_characters": len(character_files),
                "key_scenes": len(scene_files),
                "backgrounds": len(background_files),
                "files_generated": {
                    "character_profiles": character_files,
                    "scene_profiles": scene_files,
                    "background_images": background_files
                },
                "total_files": len(character_files) + len(scene_files) + len(background_files)
            }
            
            # Save master summary
            summary_file = self._save_master_summary(result)
            result["master_summary"] = summary_file
            
            logger.info(f"✅ Smart processing complete! Generated {result['total_files']} high-quality files")
            return result
            
        except Exception as e:
            logger.error(f"Smart processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_book_structure(self, text: str, title: str, author: str) -> Dict[str, Any]:
        """Analyze book structure efficiently"""
        
        word_count = len(text.split())
        
        return {
            "title": title,
            "author": author,
            "word_count": word_count,
            "estimated_reading_time": word_count // 200,
            "genre": self._detect_genre(text),
            "setting": self._detect_main_setting(text),
            "themes": self._detect_main_themes(text),
            "processed_at": datetime.now().isoformat()
        }
    
    def _extract_main_characters(self, text: str, max_count: int) -> List[Dict[str, Any]]:
        """Extract only the main characters, not every proper noun"""
        
        # Find proper nouns
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Much more restrictive filtering
        stop_words = {
            'Chapter', 'The', 'And', 'But', 'Or', 'So', 'Then', 'Now', 'Here', 'There',
            'This', 'That', 'These', 'Those', 'When', 'Where', 'Why', 'How', 'What',
            'England', 'London', 'France', 'Church', 'Cathedral', 'Father', 'Mother',
            'Brother', 'Sister', 'Mister', 'Mrs', 'Miss', 'Doctor', 'Captain',
            'After', 'His', 'However', 'With', 'All', 'From', 'Earth', 'While',
            'Its', 'Could', 'Who', 'Black', 'Right', 'Hey', 'Calm', 'Hehe',
            'Subconsciously', 'Between'  # These were showing up as "characters"
        }
        
        # Count occurrences with much higher threshold
        name_counts = {}
        for name in potential_names:
            if name not in stop_words and len(name) > 2:
                name_counts[name] = name_counts.get(name, 0) + 1
        
        # Only keep names mentioned frequently (likely real characters)
        character_names = [name for name, count in name_counts.items() if count >= 10]
        
        # Sort by frequency and take top characters
        character_names.sort(key=lambda x: name_counts[x], reverse=True)
        main_characters = character_names[:max_count]
        
        # Create character profiles
        characters = []
        for name in main_characters:
            character = {
                "name": name,
                "mentions": name_counts[name],
                "description": self._get_character_description(name, text),
                "personality": self._analyze_personality(name, text),
                "age": self._estimate_age(name, text),
                "role": "main" if name_counts[name] >= 20 else "supporting"
            }
            characters.append(character)
        
        logger.info(f"Found {len(characters)} main characters: {[c['name'] for c in characters]}")
        return characters
    
    def _identify_key_scenes(self, text: str, characters: List[Dict], max_scenes: int) -> List[Dict[str, Any]]:
        """Identify key dramatic scenes, not every paragraph"""
        
        # Split text into larger chunks (not tiny segments)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        
        # Score paragraphs for "importance"
        scored_scenes = []
        
        for i, paragraph in enumerate(paragraphs):
            score = self._score_scene_importance(paragraph, characters)
            
            if score > 3:  # Only scenes with high importance score
                scene = {
                    "index": i,
                    "description": paragraph[:300],  # First 300 chars
                    "characters": self._find_characters_in_text(paragraph, characters),
                    "setting": self._detect_scene_setting(paragraph),
                    "mood": self._detect_scene_mood(paragraph),
                    "score": score
                }
                scored_scenes.append(scene)
        
        # Sort by score and take top scenes
        scored_scenes.sort(key=lambda x: x['score'], reverse=True)
        key_scenes = scored_scenes[:max_scenes]
        
        logger.info(f"Identified {len(key_scenes)} key scenes")
        return key_scenes
    
    def _score_scene_importance(self, text: str, characters: List[Dict]) -> int:
        """Score how important/dramatic a scene is"""
        
        score = 0
        text_lower = text.lower()
        
        # Dialogue increases importance
        if '"' in text:
            score += 2
        
        # Multiple characters = more important
        char_count = sum(1 for char in characters if char['name'].lower() in text_lower)
        score += char_count
        
        # Action words
        action_words = ['said', 'walked', 'ran', 'looked', 'found', 'discovered', 'realized']
        score += sum(1 for word in action_words if word in text_lower)
        
        # Emotional words
        emotion_words = ['smiled', 'angry', 'surprised', 'worried', 'excited', 'afraid']
        score += sum(1 for word in emotion_words if word in text_lower)
        
        # Length bonus (substantial scenes)
        if len(text) > 200:
            score += 1
        
        return score
    
    def _extract_main_settings(self, text: str) -> List[str]:
        """Extract main settings from the book"""
        
        text_lower = text.lower()
        
        settings = []
        setting_indicators = {
            'marketplace': ['market', 'stall', 'vendor', 'crowd'],
            'cathedral': ['cathedral', 'church', 'priest'],
            'London street': ['london', 'street'],
            'village': ['village', 'cottage'],
            'forest': ['forest', 'tree', 'wood'],
            'castle': ['castle', 'tower']
        }
        
        for setting, keywords in setting_indicators.items():
            if sum(1 for keyword in keywords if keyword in text_lower) >= 2:
                settings.append(setting)
        
        return settings[:4]  # Max 4 main settings
    
    def _get_character_description(self, name: str, text: str) -> str:
        """Get a good description of the character"""
        
        # Find sentences containing the character name
        sentences = text.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if name in sentence and len(sentence.strip()) > 30:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:2]) + '.'
        else:
            return f"A character named {name} who plays an important role in the story."
    
    def _analyze_personality(self, name: str, text: str) -> List[str]:
        """Analyze character personality"""
        
        # Get context around character mentions
        character_text = self._get_character_context(name, text, 200)
        text_lower = character_text.lower()
        
        traits = []
        trait_indicators = {
            'kind': ['kind', 'gentle', 'caring', 'helping', 'good'],
            'brave': ['brave', 'courage', 'bold', 'fearless'],
            'intelligent': ['smart', 'clever', 'wise', 'intelligent'],
            'cheerful': ['smiled', 'laughing', 'happy', 'cheerful'],
            'serious': ['serious', 'stern', 'grave', 'solemn'],
            'mysterious': ['mysterious', 'secretive', 'enigmatic']
        }
        
        for trait, indicators in trait_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                traits.append(trait)
        
        return traits[:3] if traits else ['unknown']
    
    def _get_character_context(self, name: str, text: str, context_size: int) -> str:
        """Get text context around character mentions"""
        
        contexts = []
        name_lower = name.lower()
        words = text.split()
        
        for i, word in enumerate(words):
            if name_lower in word.lower():
                start = max(0, i - context_size // 2)
                end = min(len(words), i + context_size // 2)
                context = ' '.join(words[start:end])
                contexts.append(context)
                
                if len(contexts) >= 3:  # Limit context
                    break
        
        return ' '.join(contexts)
    
    def _estimate_age(self, name: str, text: str) -> str:
        """Estimate character age"""
        
        context = self._get_character_context(name, text, 100).lower()
        
        if any(word in context for word in ['young', 'child', 'boy', 'girl']):
            return 'young'
        elif any(word in context for word in ['old', 'elderly', 'aged']):
            return 'elderly'
        else:
            return 'adult'
    
    def _find_characters_in_text(self, text: str, characters: List[Dict]) -> List[str]:
        """Find which characters appear in a text"""
        
        text_lower = text.lower()
        found_characters = []
        
        for char in characters:
            if char['name'].lower() in text_lower:
                found_characters.append(char['name'])
        
        return found_characters
    
    def _detect_scene_setting(self, text: str) -> str:
        """Detect setting of a scene"""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['market', 'stall', 'vendor']):
            return 'marketplace'
        elif any(word in text_lower for word in ['cathedral', 'church']):
            return 'cathedral'
        elif any(word in text_lower for word in ['street', 'road', 'london']):
            return 'street'
        elif any(word in text_lower for word in ['home', 'house', 'room']):
            return 'indoor'
        else:
            return 'outdoor scene'
    
    def _detect_scene_mood(self, text: str) -> str:
        """Detect mood of a scene"""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['smiled', 'happy', 'cheerful']):
            return 'cheerful'
        elif any(word in text_lower for word in ['worried', 'anxious', 'fear']):
            return 'tense'
        elif any(word in text_lower for word in ['mysterious', 'secret', 'hidden']):
            return 'mysterious'
        else:
            return 'neutral'
    
    def _detect_genre(self, text: str) -> str:
        """Detect book genre"""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['mystery', 'detective', 'secret']):
            return 'mystery'
        elif any(word in text_lower for word in ['magic', 'wizard', 'fantasy']):
            return 'fantasy'
        elif any(word in text_lower for word in ['love', 'romance', 'heart']):
            return 'romance'
        else:
            return 'literary fiction'
    
    def _detect_main_setting(self, text: str) -> str:
        """Detect main setting of the book"""
        
        text_lower = text.lower()
        
        if 'london' in text_lower:
            return 'London, England'
        elif any(word in text_lower for word in ['village', 'countryside']):
            return 'Village'
        elif any(word in text_lower for word in ['castle', 'medieval']):
            return 'Medieval setting'
        else:
            return 'Unknown setting'
    
    def _detect_main_themes(self, text: str) -> List[str]:
        """Detect main themes"""
        
        text_lower = text.lower()
        themes = []
        
        if any(word in text_lower for word in ['family', 'father', 'mother']):
            themes.append('family')
        if any(word in text_lower for word in ['friend', 'friendship']):
            themes.append('friendship')
        if any(word in text_lower for word in ['mystery', 'secret']):
            themes.append('mystery')
        
        return themes[:3]
    
    def _save_character_profile(self, character: Dict[str, Any]) -> str:
        """Save character profile to JSON"""
        
        filename = f"{character['name'].lower().replace(' ', '_')}_profile.json"
        filepath = self.json_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(character, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _save_scene_profile(self, scene: Dict[str, Any], scene_num: int) -> str:
        """Save scene profile to JSON"""
        
        filename = f"scene_{scene_num:02d}_profile.json"
        filepath = self.json_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scene, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _save_master_summary(self, result: Dict[str, Any]) -> str:
        """Save master summary file"""
        
        filepath = self.json_dir / "book_summary.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return str(filepath)


def main():
    """Test the enhanced system"""
    
    processor = SmartBookProcessor()
    
    # Test with sample text
    sample_text = '''Emma walked through the busy marketplace in London, her basket clutched tightly in her hand. The morning sun cast long shadows between the wooden stalls.

"Fresh apples! Best in all of England!" called out Thomas, the fruit vendor, waving to catch her attention.

Emma smiled and approached his stall. "Good morning, Thomas. How much for a dozen?"

"For you, my dear Emma, just two shillings," Thomas replied with a grin. "Your father was always good to me."

James was nowhere to be found, but Emma continued her search through the bustling market. The mysterious letter in her pocket weighed heavily on her mind.'''
    
    result = processor.process_book_smart(sample_text, "Test Novel", "Test Author", max_characters=3, max_scenes=2)
    
    if result["success"]:
        print("✅ Enhanced processing complete!")
        print(f"📊 Generated {result['total_files']} high-quality files")
        print(f"👥 Main characters: {result['main_characters']}")
        print(f"🎬 Key scenes: {result['key_scenes']}")
        print(f"🏞️  Backgrounds: {result['backgrounds']}")
    else:
        print(f"❌ Processing failed: {result['error']}")


if __name__ == "__main__":
    main()