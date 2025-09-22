#!/usr/bin/env python3
"""
Comprehensive JSON Generator for EchoTales
Generates complete JSON files with character profiles, images, and animations
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
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ImageGenerator:
    """Handles free image generation using various APIs"""
    
    def __init__(self):
        self.apis = [
            {"name": "Picsum", "type": "random_photo"},
            {"name": "placeholder", "type": "placeholder"},
            {"name": "generated", "type": "character_art"}
        ]
        self.base_urls = {
            "picsum": "https://picsum.photos",
            "placeholder": "https://via.placeholder.com",
            "unsplash": "https://source.unsplash.com"
        }
    
    def generate_character_image(self, character_name: str, description: str, output_dir: Path) -> str:
        """Generate a character image using free APIs"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{character_name.lower().replace(' ', '_')}_character.jpg"
        filepath = output_dir / filename
        
        # Try different image generation strategies
        success = False
        
        # Strategy 1: Use character-based search terms
        if not success:
            success = self._try_unsplash_character(character_name, description, filepath)
        
        # Strategy 2: Use placeholder with character info
        if not success:
            success = self._try_placeholder_character(character_name, description, filepath)
        
        # Strategy 3: Generate a simple colored placeholder
        if not success:
            success = self._generate_simple_placeholder(character_name, filepath)
        
        return str(filepath) if success else None
    
    def generate_scene_image(self, scene_description: str, characters: List[str], setting: str, output_dir: Path) -> str:
        """Generate a scene image"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from scene info
        scene_words = scene_description.split()[:3]
        filename = f"scene_{'_'.join(scene_words).lower()}.jpg"
        filepath = output_dir / filename
        
        # Try to generate scene image
        success = False
        
        # Strategy 1: Setting-based image
        if setting and not success:
            success = self._try_unsplash_setting(setting, filepath)
        
        # Strategy 2: Generic scene image
        if not success:
            success = self._try_generic_scene(filepath)
        
        return str(filepath) if success else None
    
    def _try_unsplash_character(self, character_name: str, description: str, filepath: Path) -> bool:
        """Try to get character image from Unsplash"""
        
        try:
            # Extract keywords from description
            keywords = self._extract_keywords_from_description(description)
            search_terms = ["portrait", "person"] + keywords[:2]
            query = ",".join(search_terms)
            
            url = f"{self.base_urls['unsplash']}/400x600/?{query}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Generated character image: {filepath.name}")
                return True
        except Exception as e:
            logger.warning(f"Unsplash character generation failed: {e}")
        
        return False
    
    def _try_unsplash_setting(self, setting: str, filepath: Path) -> bool:
        """Try to get setting image from Unsplash"""
        
        try:
            # Clean up setting description
            setting_clean = setting.lower().replace("the ", "").replace("a ", "")
            url = f"{self.base_urls['unsplash']}/800x600/?{setting_clean}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Generated setting image: {filepath.name}")
                return True
        except Exception as e:
            logger.warning(f"Unsplash setting generation failed: {e}")
        
        return False
    
    def _try_placeholder_character(self, character_name: str, description: str, filepath: Path) -> bool:
        """Generate placeholder character image"""
        
        try:
            # Generate a colored placeholder based on character name
            color = self._name_to_color(character_name)
            text = character_name[:10]  # Limit text length
            
            url = f"{self.base_urls['placeholder']}/400x600/{color}/ffffff?text={text}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Generated placeholder image: {filepath.name}")
                return True
        except Exception as e:
            logger.warning(f"Placeholder generation failed: {e}")
        
        return False
    
    def _try_generic_scene(self, filepath: Path) -> bool:
        """Generate generic scene image"""
        
        try:
            # Use random nature/landscape image
            scene_types = ["landscape", "nature", "architecture", "city", "forest", "mountain"]
            scene_type = random.choice(scene_types)
            
            url = f"{self.base_urls['unsplash']}/800x600/?{scene_type}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Generated generic scene: {filepath.name}")
                return True
        except Exception as e:
            logger.warning(f"Generic scene generation failed: {e}")
        
        return False
    
    def _generate_simple_placeholder(self, character_name: str, filepath: Path) -> bool:
        """Generate a simple colored rectangle as fallback"""
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a simple colored image
            width, height = 400, 600
            color = self._name_to_rgb_color(character_name)
            
            img = Image.new('RGB', (width, height), color)
            draw = ImageDraw.Draw(img)
            
            # Add character name
            try:
                # Try to use a font
                font = ImageFont.load_default()
                text = character_name
                
                # Calculate text position
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                
                # Draw text with contrasting color
                text_color = (255, 255, 255) if sum(color) < 384 else (0, 0, 0)
                draw.text((x, y), text, fill=text_color, font=font)
                
            except Exception:
                # If font fails, just create colored rectangle
                pass
            
            img.save(filepath, 'JPEG')
            logger.info(f"Generated simple placeholder: {filepath.name}")
            return True
            
        except ImportError:
            logger.warning("PIL not available for simple placeholder generation")
        except Exception as e:
            logger.warning(f"Simple placeholder generation failed: {e}")
        
        return False
    
    def _extract_keywords_from_description(self, description: str) -> List[str]:
        """Extract relevant keywords from character description"""
        
        keywords = []
        
        # Common appearance keywords
        appearance_keywords = [
            'tall', 'short', 'young', 'old', 'blonde', 'brunette', 'redhead',
            'blue eyes', 'green eyes', 'brown eyes', 'beard', 'mustache',
            'elegant', 'rugged', 'pretty', 'handsome', 'beautiful'
        ]
        
        description_lower = description.lower()
        for keyword in appearance_keywords:
            if keyword in description_lower:
                keywords.append(keyword.replace(' ', '+'))
        
        return keywords[:3]  # Limit to 3 keywords
    
    def _name_to_color(self, name: str) -> str:
        """Convert name to hex color for placeholders"""
        
        # Simple hash-based color generation
        hash_value = hash(name) % 16777216  # 24-bit color space
        return f"{hash_value:06x}"
    
    def _name_to_rgb_color(self, name: str) -> tuple:
        """Convert name to RGB color"""
        
        hash_value = hash(name)
        r = (hash_value & 0xFF0000) >> 16
        g = (hash_value & 0x00FF00) >> 8
        b = hash_value & 0x0000FF
        
        # Ensure colors are not too dark
        r = max(r, 64)
        g = max(g, 64)
        b = max(b, 64)
        
        return (r, g, b)


class AnimationGenerator:
    """Creates pseudo-animations for characters"""
    
    def generate_character_animation(self, character_info: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Generate animation data for character"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        character_name = character_info.get('name', 'Unknown')
        personality = character_info.get('personality_traits', [])
        
        # Create animation profile
        animation_data = {
            "character_name": character_name,
            "animation_type": self._determine_animation_type(personality),
            "movements": self._generate_movement_patterns(personality),
            "expressions": self._generate_expressions(personality),
            "voice_characteristics": self._generate_voice_profile(character_info),
            "animation_frames": self._create_frame_sequence(),
            "generated_at": datetime.now().isoformat()
        }
        
        # Save animation data
        filename = f"{character_name.lower().replace(' ', '_')}_animation.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(animation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated animation profile: {filepath.name}")
        return animation_data
    
    def _determine_animation_type(self, personality: List[str]) -> str:
        """Determine character animation type based on personality"""
        
        if any(trait in personality for trait in ['energetic', 'cheerful', 'optimistic']):
            return "bouncy"
        elif any(trait in personality for trait in ['calm', 'serene', 'peaceful']):
            return "gentle"
        elif any(trait in personality for trait in ['mysterious', 'secretive', 'enigmatic']):
            return "smooth"
        elif any(trait in personality for trait in ['nervous', 'anxious', 'fidgety']):
            return "jittery"
        else:
            return "neutral"
    
    def _generate_movement_patterns(self, personality: List[str]) -> List[Dict[str, Any]]:
        """Generate movement patterns"""
        
        base_movements = [
            {"type": "idle", "duration": 3.0, "description": "Standing naturally"},
            {"type": "gesture", "duration": 1.5, "description": "Hand gestures while speaking"},
            {"type": "walk", "duration": 2.0, "description": "Walking movement"},
            {"type": "turn", "duration": 1.0, "description": "Turning to face different directions"}
        ]
        
        # Modify based on personality
        if 'energetic' in personality:
            for movement in base_movements:
                movement['intensity'] = 'high'
                movement['speed_multiplier'] = 1.5
        elif 'calm' in personality:
            for movement in base_movements:
                movement['intensity'] = 'low'
                movement['speed_multiplier'] = 0.8
        else:
            for movement in base_movements:
                movement['intensity'] = 'medium'
                movement['speed_multiplier'] = 1.0
        
        return base_movements
    
    def _generate_expressions(self, personality: List[str]) -> List[Dict[str, Any]]:
        """Generate facial expressions"""
        
        base_expressions = ["neutral", "speaking", "listening", "thinking"]
        
        # Add personality-specific expressions
        if 'cheerful' in personality:
            base_expressions.extend(["smiling", "laughing", "excited"])
        if 'serious' in personality:
            base_expressions.extend(["focused", "stern", "contemplative"])
        if 'mysterious' in personality:
            base_expressions.extend(["knowing_smile", "suspicious", "secretive"])
        
        expressions = []
        for expr in base_expressions:
            expressions.append({
                "expression": expr,
                "duration": random.uniform(1.0, 3.0),
                "intensity": random.choice(["subtle", "moderate", "strong"])
            })
        
        return expressions
    
    def _generate_voice_profile(self, character_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate voice characteristics"""
        
        age = character_info.get('age', 'Unknown')
        personality = character_info.get('personality_traits', [])
        
        voice_profile = {
            "pitch": "medium",
            "speed": "normal",
            "tone": "neutral",
            "accent": "standard"
        }
        
        # Adjust based on age
        if isinstance(age, str) and 'young' in age.lower():
            voice_profile['pitch'] = "higher"
        elif isinstance(age, str) and 'old' in age.lower():
            voice_profile['pitch'] = "lower"
        
        # Adjust based on personality
        if 'cheerful' in personality:
            voice_profile['tone'] = "bright"
            voice_profile['speed'] = "fast"
        elif 'serious' in personality:
            voice_profile['tone'] = "grave"
            voice_profile['speed'] = "slow"
        elif 'mysterious' in personality:
            voice_profile['tone'] = "whispered"
            voice_profile['speed'] = "slow"
        
        return voice_profile
    
    def _create_frame_sequence(self) -> List[Dict[str, Any]]:
        """Create animation frame sequence"""
        
        frames = []
        frame_count = 30  # 30 frames for a 1-second loop at 30fps
        
        for i in range(frame_count):
            frame = {
                "frame_number": i,
                "timestamp": i / 30.0,  # Time in seconds
                "position_x": 0,  # Base position
                "position_y": 0,
                "rotation": 0,
                "scale": 1.0,
                "opacity": 1.0
            }
            
            # Add subtle breathing animation
            breathing_offset = 2 * (i % 30) / 30.0  # Breathe cycle
            frame['position_y'] = round(2 * abs(breathing_offset - 1), 2)
            
            frames.append(frame)
        
        return frames


class ComprehensiveJSONGenerator:
    """Main class for generating comprehensive JSON files"""
    
    def __init__(self):
        self.image_generator = ImageGenerator()
        self.animation_generator = AnimationGenerator()
        
        # Create output directories
        self.output_dir = Path("data/generated")
        self.images_dir = self.output_dir / "images"
        self.json_dir = self.output_dir / "json"
        self.scenes_dir = self.output_dir / "scenes"
        self.animations_dir = self.output_dir / "animations"
        
        # Create directories
        for dir_path in [self.output_dir, self.images_dir, self.json_dir, self.scenes_dir, self.animations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_novel(self, novel_text: str, title: str = "Unknown Title", author: str = "Unknown Author") -> Dict[str, Any]:
        """Process an entire novel and generate comprehensive JSON files"""
        
        logger.info(f"Starting comprehensive processing of '{title}' by {author}")
        
        try:
            # Analyze the novel
            novel_analysis = self._analyze_novel_structure(novel_text, title, author)
            
            # Extract characters
            characters = self._extract_characters(novel_text)
            
            # Extract chapters
            chapters = self._extract_chapters(novel_text)
            
            # Generate character profiles with images and animations
            character_files = []
            for character in characters:
                char_profile = self._create_character_profile(character, novel_text)
                char_file = self._save_character_json(char_profile)
                character_files.append(char_file)
            
            # Generate chapter files
            chapter_files = []
            for chapter in chapters:
                chapter_profile = self._create_chapter_profile(chapter, characters)
                chapter_file = self._save_chapter_json(chapter_profile)
                chapter_files.append(chapter_file)
            
            # Generate background/setting images
            background_files = self._generate_background_images(novel_text, chapters)
            
            # Create master index
            master_index = self._create_master_index(novel_analysis, character_files, chapter_files, background_files)
            master_file = self._save_master_index(master_index)
            
            result = {
                "success": True,
                "novel_info": novel_analysis,
                "generated_files": {
                    "characters": character_files,
                    "chapters": chapter_files,
                    "backgrounds": background_files,
                    "master_index": master_file
                },
                "processing_time": time.time(),
                "total_files_generated": len(character_files) + len(chapter_files) + len(background_files) + 1
            }
            
            logger.info(f"Successfully processed '{title}' - Generated {result['total_files_generated']} files")
            return result
            
        except Exception as e:
            logger.error(f"Error processing novel: {e}")
            return {
                "success": False,
                "error": str(e),
                "novel_info": {"title": title, "author": author}
            }
    
    def _analyze_novel_structure(self, novel_text: str, title: str, author: str) -> Dict[str, Any]:
        """Analyze overall novel structure"""
        
        word_count = len(novel_text.split())
        chapter_count = len(re.findall(r'\[CHAPTER_START_\d+\]', novel_text))
        
        if chapter_count == 0:
            chapter_count = len(re.findall(r'Chapter \d+', novel_text))
        if chapter_count == 0:
            chapter_count = 1
        
        return {
            "title": title,
            "author": author,
            "total_words": word_count,
            "total_chapters": chapter_count,
            "estimated_reading_time": round(word_count / 200),  # 200 words per minute
            "genre": self._detect_genre(novel_text),
            "main_themes": self._extract_themes(novel_text),
            "setting": self._detect_setting(novel_text),
            "time_period": self._detect_time_period(novel_text),
            "processed_at": datetime.now().isoformat(),
            "booknlp_used": False  # Set to False since we're using fallback analysis
        }
    
    def _extract_characters(self, novel_text: str) -> List[Dict[str, Any]]:
        """Extract characters from the novel text"""
        
        # Find proper nouns that might be character names
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', novel_text)
        
        # Filter out common non-name words
        stop_words = {
            'Chapter', 'The', 'And', 'But', 'Or', 'So', 'Then', 'Now', 'Here', 'There',
            'This', 'That', 'These', 'Those', 'When', 'Where', 'Why', 'How', 'What',
            'England', 'London', 'France', 'Church', 'Cathedral', 'Father', 'Mother',
            'Brother', 'Sister', 'Mister', 'Mrs', 'Miss', 'Doctor', 'Captain'
        }
        
        # Count occurrences and filter
        name_counts = {}
        for name in potential_names:
            if name not in stop_words and len(name) > 2:
                name_counts[name] = name_counts.get(name, 0) + 1
        
        # Keep names that appear at least 3 times
        character_names = [name for name, count in name_counts.items() if count >= 3]
        
        # Create character objects
        characters = []
        for name in character_names[:20]:  # Limit to top 20 characters
            character = {
                "name": name,
                "mentions": name_counts[name],
                "description": self._extract_character_description(name, novel_text),
                "role": self._determine_character_role(name, name_counts[name], len(character_names))
            }
            characters.append(character)
        
        # Sort by mention count (main characters first)
        characters.sort(key=lambda x: x['mentions'], reverse=True)
        
        return characters
    
    def _extract_chapters(self, novel_text: str) -> List[Dict[str, Any]]:
        """Extract chapters from the novel"""
        
        chapters = []
        
        # Split by chapter markers
        chapter_pattern = r'\[CHAPTER_START_(\d+)\]'
        parts = re.split(chapter_pattern, novel_text)
        
        if len(parts) <= 1:
            # No chapter markers, treat as single chapter
            chapters.append({
                "chapter_number": 1,
                "title": "Chapter 1",
                "content": novel_text.strip(),
                "word_count": len(novel_text.split()),
                "summary": self._generate_chapter_summary(novel_text)
            })
        else:
            # Process marked chapters
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    chapter_num = int(parts[i])
                    content = parts[i + 1].strip()
                    
                    if content:
                        chapters.append({
                            "chapter_number": len(chapters) + 1,
                            "title": f"Chapter {len(chapters) + 1}",
                            "content": content,
                            "word_count": len(content.split()),
                            "summary": self._generate_chapter_summary(content),
                            "page_number": chapter_num
                        })
        
        return chapters
    
    def _create_character_profile(self, character_data: Dict[str, Any], novel_text: str) -> Dict[str, Any]:
        """Create comprehensive character profile"""
        
        character_name = character_data['name']
        
        # Generate character image
        description = character_data.get('description', f"Character named {character_name}")
        image_path = self.image_generator.generate_character_image(
            character_name, description, self.images_dir
        )
        
        # Extract additional character information
        personality_traits = self._analyze_personality(character_name, novel_text)
        relationships = self._analyze_relationships(character_name, novel_text)
        character_arc = self._analyze_character_arc(character_name, novel_text)
        
        # Create comprehensive profile
        profile = {
            "name": character_name,
            "role": character_data.get('role', 'supporting'),
            "description": description,
            "personality_traits": personality_traits,
            "physical_appearance": self._extract_physical_appearance(character_name, novel_text),
            "age": self._estimate_age(character_name, novel_text),
            "relationships": relationships,
            "character_arc": character_arc,
            "important_scenes": self._find_important_scenes(character_name, novel_text),
            "memorable_quotes": self._extract_quotes(character_name, novel_text),
            "mentions_count": character_data.get('mentions', 0),
            "image_path": image_path,
            "generated_at": datetime.now().isoformat()
        }
        
        # Generate animation data
        animation_data = self.animation_generator.generate_character_animation(
            profile, self.animations_dir
        )
        profile["animation_profile"] = animation_data
        
        return profile
    
    def _create_chapter_profile(self, chapter_data: Dict[str, Any], characters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive chapter profile"""
        
        content = chapter_data['content']
        chapter_num = chapter_data['chapter_number']
        
        # Find characters in this chapter
        chapter_characters = []
        for char in characters:
            if char['name'] in content:
                chapter_characters.append(char['name'])
        
        # Generate scene image
        scene_desc = content[:200]  # First 200 characters as scene description
        setting = self._detect_chapter_setting(content)
        scene_image = self.image_generator.generate_scene_image(
            scene_desc, chapter_characters, setting, self.scenes_dir
        )
        
        # Analyze chapter
        profile = {
            "chapter_number": chapter_num,
            "title": chapter_data.get('title', f"Chapter {chapter_num}"),
            "summary": chapter_data.get('summary', ''),
            "word_count": chapter_data.get('word_count', 0),
            "characters_present": chapter_characters,
            "setting": setting,
            "mood": self._analyze_chapter_mood(content),
            "key_events": self._extract_key_events(content),
            "themes": self._extract_chapter_themes(content),
            "scene_image": scene_image,
            "reading_time_minutes": round(chapter_data.get('word_count', 0) / 200),
            "generated_at": datetime.now().isoformat()
        }
        
        return profile
    
    def _generate_background_images(self, novel_text: str, chapters: List[Dict[str, Any]]) -> List[str]:
        """Generate background/setting images"""
        
        background_files = []
        
        # Extract unique settings
        settings = set()
        settings.add(self._detect_setting(novel_text))  # Main setting
        
        for chapter in chapters:
            chapter_setting = self._detect_chapter_setting(chapter.get('content', ''))
            if chapter_setting:
                settings.add(chapter_setting)
        
        # Generate images for each setting
        for setting in settings:
            if setting and setting != 'unknown':
                scene_image = self.image_generator.generate_scene_image(
                    f"A view of {setting}", [], setting, self.scenes_dir
                )
                if scene_image:
                    background_files.append(scene_image)
        
        return background_files
    
    def _create_master_index(self, novel_analysis: Dict[str, Any], character_files: List[str], 
                           chapter_files: List[str], background_files: List[str]) -> Dict[str, Any]:
        """Create master index file"""
        
        return {
            "novel_metadata": novel_analysis,
            "generated_files": {
                "character_profiles": [Path(f).name for f in character_files],
                "chapter_profiles": [Path(f).name for f in chapter_files],
                "background_images": [Path(f).name for f in background_files],
                "total_files": len(character_files) + len(chapter_files) + len(background_files)
            },
            "file_paths": {
                "json_directory": str(self.json_dir),
                "images_directory": str(self.images_dir),
                "scenes_directory": str(self.scenes_dir),
                "animations_directory": str(self.animations_dir)
            },
            "generation_info": {
                "generated_at": datetime.now().isoformat(),
                "generator_version": "1.0",
                "processing_complete": True
            }
        }
    
    def _save_character_json(self, character_profile: Dict[str, Any]) -> str:
        """Save character profile to JSON file"""
        
        character_name = character_profile['name']
        filename = f"{character_name.lower().replace(' ', '_')}_profile.json"
        filepath = self.json_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(character_profile, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved character profile: {filename}")
        return str(filepath)
    
    def _save_chapter_json(self, chapter_profile: Dict[str, Any]) -> str:
        """Save chapter profile to JSON file"""
        
        chapter_num = chapter_profile['chapter_number']
        filename = f"chapter_{chapter_num:02d}_profile.json"
        filepath = self.json_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chapter_profile, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved chapter profile: {filename}")
        return str(filepath)
    
    def _save_master_index(self, master_index: Dict[str, Any]) -> str:
        """Save master index file"""
        
        filepath = self.json_dir / "master_index.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(master_index, f, indent=2, ensure_ascii=False)
        
        logger.info("Saved master index file")
        return str(filepath)
    
    # Helper methods for text analysis
    
    def _extract_character_description(self, character_name: str, text: str) -> str:
        """Extract description of a character from text"""
        
        # Look for sentences containing the character name
        sentences = text.split('.')
        descriptions = []
        
        for sentence in sentences:
            if character_name in sentence and len(sentence.strip()) > 20:
                descriptions.append(sentence.strip())
        
        # Return first few relevant sentences
        if descriptions:
            return '. '.join(descriptions[:2]) + '.'
        else:
            return f"A character named {character_name} who appears in the story."
    
    def _determine_character_role(self, name: str, mentions: int, total_characters: int) -> str:
        """Determine character role based on mentions"""
        
        if mentions >= 20:
            return "protagonist"
        elif mentions >= 10:
            return "major"
        elif mentions >= 5:
            return "supporting"
        else:
            return "minor"
    
    def _analyze_personality(self, character_name: str, text: str) -> List[str]:
        """Analyze character personality from text"""
        
        # Simple personality trait detection
        traits = []
        
        character_context = self._get_character_context(character_name, text)
        
        # Check for various personality indicators
        trait_indicators = {
            'kind': ['kind', 'gentle', 'caring', 'helping'],
            'brave': ['brave', 'courage', 'bold', 'fearless'],
            'intelligent': ['smart', 'clever', 'wise', 'intelligent'],
            'cheerful': ['smiled', 'laughing', 'happy', 'cheerful'],
            'serious': ['serious', 'stern', 'grave', 'solemn'],
            'mysterious': ['mysterious', 'secretive', 'enigmatic', 'hidden']
        }
        
        context_lower = character_context.lower()
        for trait, indicators in trait_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                traits.append(trait)
        
        return traits if traits else ['unknown']
    
    def _get_character_context(self, character_name: str, text: str, context_size: int = 100) -> str:
        """Get text context around character mentions"""
        
        contexts = []
        text_words = text.split()
        
        for i, word in enumerate(text_words):
            if character_name in word:
                start = max(0, i - context_size // 2)
                end = min(len(text_words), i + context_size // 2)
                context = ' '.join(text_words[start:end])
                contexts.append(context)
        
        return ' '.join(contexts[:3])  # Return first 3 contexts
    
    def _analyze_relationships(self, character_name: str, text: str) -> List[Dict[str, str]]:
        """Analyze character relationships"""
        
        relationships = []
        
        # Simple relationship detection based on proximity and context
        relationship_words = {
            'father': 'parent',
            'mother': 'parent', 
            'brother': 'sibling',
            'sister': 'sibling',
            'friend': 'friend',
            'husband': 'spouse',
            'wife': 'spouse'
        }
        
        context = self._get_character_context(character_name, text, 50)
        
        for rel_word, rel_type in relationship_words.items():
            if rel_word in context.lower():
                relationships.append({
                    "type": rel_type,
                    "description": f"Has {rel_word} relationship mentioned in text"
                })
        
        return relationships
    
    def _analyze_character_arc(self, character_name: str, text: str) -> str:
        """Analyze character development arc"""
        
        # Simple arc detection based on story progression
        return "Character appears throughout the story with various interactions and developments."
    
    def _extract_physical_appearance(self, character_name: str, text: str) -> str:
        """Extract physical appearance details"""
        
        context = self._get_character_context(character_name, text)
        
        # Look for appearance-related words
        appearance_words = [
            'tall', 'short', 'young', 'old', 'blonde', 'brunette', 'redhead',
            'blue eyes', 'green eyes', 'brown eyes', 'beard', 'mustache'
        ]
        
        found_traits = []
        context_lower = context.lower()
        
        for trait in appearance_words:
            if trait in context_lower:
                found_traits.append(trait)
        
        if found_traits:
            return f"Described as {', '.join(found_traits)}"
        else:
            return "Physical appearance not explicitly described"
    
    def _estimate_age(self, character_name: str, text: str) -> str:
        """Estimate character age"""
        
        context = self._get_character_context(character_name, text).lower()
        
        age_indicators = {
            'young': ['young', 'child', 'boy', 'girl', 'teenager'],
            'adult': ['man', 'woman', 'adult'],
            'elderly': ['old', 'elderly', 'aged', 'grandfather', 'grandmother']
        }
        
        for age_group, indicators in age_indicators.items():
            if any(indicator in context for indicator in indicators):
                return age_group
        
        return "unknown"
    
    def _find_important_scenes(self, character_name: str, text: str) -> List[str]:
        """Find important scenes involving the character"""
        
        sentences = text.split('.')
        important_scenes = []
        
        for sentence in sentences:
            if (character_name in sentence and 
                len(sentence.strip()) > 30 and 
                any(word in sentence.lower() for word in ['said', 'walked', 'found', 'met', 'saw'])):
                important_scenes.append(sentence.strip() + '.')
        
        return important_scenes[:3]  # Return top 3 scenes
    
    def _extract_quotes(self, character_name: str, text: str) -> List[str]:
        """Extract character quotes"""
        
        quotes = []
        
        # Look for quoted speech near character name
        quote_pattern = r'"([^"]+)"'
        quote_matches = re.findall(quote_pattern, text)
        
        # Find quotes near character mentions
        for quote in quote_matches:
            if len(quote) > 10 and len(quote) < 200:  # Reasonable quote length
                # Check if character name appears near this quote in text
                quote_pos = text.find(f'"{quote}"')
                if quote_pos != -1:
                    context_start = max(0, quote_pos - 100)
                    context_end = min(len(text), quote_pos + len(quote) + 100)
                    context = text[context_start:context_end]
                    
                    if character_name in context:
                        quotes.append(quote)
        
        return quotes[:3]  # Return top 3 quotes
    
    def _detect_genre(self, text: str) -> str:
        """Detect novel genre"""
        
        text_lower = text.lower()
        
        genre_keywords = {
            'mystery': ['mystery', 'detective', 'murder', 'clue', 'investigate'],
            'romance': ['love', 'heart', 'romance', 'kiss', 'marriage'],
            'fantasy': ['magic', 'wizard', 'dragon', 'fantasy', 'kingdom'],
            'historical': ['war', 'century', 'historical', 'ancient', 'past'],
            'adventure': ['journey', 'quest', 'adventure', 'travel', 'explore']
        }
        
        for genre, keywords in genre_keywords.items():
            if sum(1 for keyword in keywords if keyword in text_lower) >= 2:
                return genre
        
        return 'literary fiction'
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract main themes"""
        
        themes = []
        text_lower = text.lower()
        
        theme_keywords = {
            'family': ['family', 'father', 'mother', 'brother', 'sister'],
            'friendship': ['friend', 'friendship', 'companion', 'together'],
            'love': ['love', 'romance', 'heart', 'beloved'],
            'war': ['war', 'battle', 'conflict', 'fight'],
            'mystery': ['secret', 'mystery', 'hidden', 'unknown'],
            'coming_of_age': ['young', 'growing', 'learn', 'discover']
        }
        
        for theme, keywords in theme_keywords.items():
            if sum(1 for keyword in keywords if keyword in text_lower) >= 2:
                themes.append(theme)
        
        return themes if themes else ['life']
    
    def _detect_setting(self, text: str) -> str:
        """Detect main setting"""
        
        text_lower = text.lower()
        
        # Look for location names and setting indicators
        setting_indicators = {
            'london': 'London, England',
            'england': 'England',
            'france': 'France',
            'castle': 'Medieval castle',
            'village': 'Village',
            'city': 'City',
            'forest': 'Forest',
            'cathedral': 'Cathedral',
            'church': 'Church',
            'marketplace': 'Marketplace'
        }
        
        for indicator, setting in setting_indicators.items():
            if indicator in text_lower:
                return setting
        
        return 'unknown location'
    
    def _detect_time_period(self, text: str) -> str:
        """Detect time period"""
        
        text_lower = text.lower()
        
        period_indicators = {
            'medieval': ['castle', 'knight', 'medieval', 'cathedral', 'monastery'],
            'victorian': ['victorian', '1800s', 'carriage', 'gaslight'],
            'modern': ['modern', 'today', 'current', 'contemporary'],
            'historical': ['war', 'century', 'historical', 'ancient']
        }
        
        for period, indicators in period_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return period
        
        return 'unspecified'
    
    def _generate_chapter_summary(self, chapter_content: str) -> str:
        """Generate chapter summary"""
        
        # Simple summary: first sentence + last sentence
        sentences = [s.strip() for s in chapter_content.split('.') if s.strip()]
        
        if len(sentences) >= 2:
            return f"{sentences[0]}. ... {sentences[-1]}."
        elif len(sentences) == 1:
            return sentences[0] + '.'
        else:
            return "Chapter content summary."
    
    def _detect_chapter_setting(self, content: str) -> str:
        """Detect setting for specific chapter"""
        
        return self._detect_setting(content)  # Use same logic as main setting detection
    
    def _analyze_chapter_mood(self, content: str) -> str:
        """Analyze chapter mood"""
        
        content_lower = content.lower()
        
        mood_indicators = {
            'cheerful': ['smiled', 'happy', 'laughed', 'joy', 'bright'],
            'tense': ['nervous', 'worried', 'anxious', 'fear', 'danger'],
            'mysterious': ['mysterious', 'secret', 'hidden', 'strange', 'whisper'],
            'sad': ['sad', 'tears', 'crying', 'sorrow', 'grief'],
            'peaceful': ['calm', 'peaceful', 'quiet', 'serene', 'gentle']
        }
        
        for mood, indicators in mood_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                return mood
        
        return 'neutral'
    
    def _extract_key_events(self, content: str) -> List[str]:
        """Extract key events from chapter"""
        
        # Look for action-oriented sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        key_events = []
        
        action_words = ['walked', 'ran', 'found', 'met', 'said', 'saw', 'heard', 'discovered']
        
        for sentence in sentences:
            if any(action in sentence.lower() for action in action_words) and len(sentence) > 20:
                key_events.append(sentence + '.')
        
        return key_events[:5]  # Return top 5 events
    
    def _extract_chapter_themes(self, content: str) -> List[str]:
        """Extract themes from specific chapter"""
        
        return self._extract_themes(content)  # Use same logic as main theme extraction


def main():
    """Main function for testing"""
    
    generator = ComprehensiveJSONGenerator()
    
    # Test with sample text
    sample_text = '''[CHAPTER_START_1]
Emma walked through the busy marketplace in London, her basket clutched tightly in her hand. The morning sun cast long shadows between the wooden stalls.

"Fresh apples! Best in all of England!" called out Thomas, the fruit vendor, waving to catch her attention.

Emma smiled and approached his stall. "Good morning, Thomas. How much for a dozen?"

[CHAPTER_START_2]
Near the cathedral, James was meeting with Father McKenzie about the village's ancient history.
'''
    
    result = generator.process_novel(sample_text, "Test Novel", "Test Author")
    print(f"Processing result: {result['success']}")
    if result['success']:
        print(f"Generated {result['total_files_generated']} files")


if __name__ == "__main__":
    main()