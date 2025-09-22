#!/usr/bin/env python3
"""
Chapter Segment Processor for EchoTales
Generates background settings and images for every 5-6 lines of text
Provides character information and scene context for each segment
"""

import os
import sys
import json
import re
import logging
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SceneImageGenerator:
    """Generate scene images for text segments with character context"""
    
    def __init__(self):
        self.output_dir = Path("data/generated/scenes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Free image generation APIs
        self.api_base = "https://image.pollinations.ai/prompt/"
        
    def generate_scene_image(self, segment_info: Dict[str, Any]) -> str:
        """Generate scene image based on segment information"""
        
        segment_id = segment_info["segment_id"]
        text_content = segment_info["text"]
        characters = segment_info["characters_present"]
        setting = segment_info["detected_setting"]
        
        logger.info(f"Generating scene image for segment {segment_id}")
        
        # Create detailed scene prompt
        prompt = self._create_scene_prompt(text_content, characters, setting)
        
        try:
            # Generate image using Pollinations AI
            image_path = self._request_scene_image(prompt, segment_id)
            
            if image_path:
                logger.info(f"✅ Generated scene image: {image_path}")
                return image_path
            
        except Exception as e:
            logger.warning(f"Scene image generation failed for segment {segment_id}: {e}")
        
        # Fallback to placeholder
        return self._create_scene_placeholder(segment_id, setting)
    
    def _create_scene_prompt(self, text: str, characters: List[str], setting: str) -> str:
        """Create detailed prompt for scene image generation"""
        
        # Extract key elements from text
        mood = self._detect_mood(text)
        action = self._detect_action(text)
        time_of_day = self._detect_time(text)
        
        # Build character description
        character_desc = ""
        if characters:
            if len(characters) == 1:
                character_desc = f"featuring {characters[0]}"
            elif len(characters) == 2:
                character_desc = f"with {characters[0]} and {characters[1]}"
            else:
                character_desc = f"with {', '.join(characters[:-1])} and {characters[-1]}"
        
        # Build comprehensive prompt
        prompt_parts = [
            f"{setting} scene",
            character_desc,
            f"{mood} atmosphere",
            f"{time_of_day} lighting",
            action,
            "detailed illustration, cinematic composition, high quality"
        ]
        
        prompt = ", ".join([part for part in prompt_parts if part])
        
        # Clean and limit prompt length
        prompt = re.sub(r'[^\w\s,.-]', '', prompt)
        prompt = prompt[:500]  # Limit prompt length
        
        logger.info(f"Scene prompt: {prompt[:100]}...")
        return prompt
    
    def _detect_mood(self, text: str) -> str:
        """Detect mood from text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['smiled', 'laughed', 'bright', 'cheerful', 'happy']):
            return "cheerful, bright"
        elif any(word in text_lower for word in ['dark', 'shadow', 'fear', 'worried', 'anxious']):
            return "dark, mysterious"
        elif any(word in text_lower for word in ['peaceful', 'calm', 'gentle', 'quiet']):
            return "peaceful, serene"
        elif any(word in text_lower for word in ['busy', 'bustling', 'crowd', 'noise']):
            return "bustling, energetic"
        else:
            return "atmospheric"
    
    def _detect_action(self, text: str) -> str:
        """Detect main action from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['walked', 'walking', 'approached', 'moving']):
            return "people walking"
        elif any(word in text_lower for word in ['talking', 'conversation', 'speaking', 'said']):
            return "characters in conversation"
        elif any(word in text_lower for word in ['market', 'selling', 'vendor', 'stall']):
            return "marketplace activity"
        elif any(word in text_lower for word in ['looking', 'watching', 'observing']):
            return "characters observing"
        else:
            return "scene activity"
    
    def _detect_time(self, text: str) -> str:
        """Detect time of day from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['morning', 'dawn', 'sunrise']):
            return "morning"
        elif any(word in text_lower for word in ['noon', 'midday', 'bright sun']):
            return "midday"
        elif any(word in text_lower for word in ['evening', 'sunset', 'dusk']):
            return "evening"
        elif any(word in text_lower for word in ['night', 'dark', 'moon']):
            return "nighttime"
        else:
            return "natural daylight"
    
    def _request_scene_image(self, prompt: str, segment_id: str) -> Optional[str]:
        """Request scene image from API"""
        
        try:
            # Clean prompt for URL
            clean_prompt = prompt.replace(" ", "%20").replace(",", "%2C").replace("&", "%26")
            url = f"{self.api_base}{clean_prompt}"
            
            # Make request
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Save image
                image_path = self.output_dir / f"segment_{segment_id}_scene.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify image
                if image_path.exists() and image_path.stat().st_size > 1000:
                    return str(image_path)
                else:
                    logger.warning(f"Generated image too small: {image_path.stat().st_size} bytes")
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
        
        return None
    
    def _create_scene_placeholder(self, segment_id: str, setting: str) -> str:
        """Create placeholder scene image"""
        
        placeholder_dir = self.output_dir / "placeholders"
        placeholder_dir.mkdir(exist_ok=True)
        
        # Generate placeholder with scene info
        try:
            color_hash = abs(hash(f"{segment_id}_{setting}")) % 255
            color = f"{color_hash:02x}{(color_hash * 2) % 255:02x}{(color_hash * 3) % 255:02x}"
            
            url = f"https://via.placeholder.com/800x600/{color}/FFFFFF?text=Scene+{segment_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                placeholder_path = placeholder_dir / f"segment_{segment_id}_placeholder.png"
                with open(placeholder_path, 'wb') as f:
                    f.write(response.content)
                return str(placeholder_path)
                
        except:
            pass
        
        return str(placeholder_dir / f"default_scene_{segment_id}.jpg")


class ChapterSegmentProcessor:
    """Main processor for chapter segments with character context"""
    
    def __init__(self):
        self.output_dir = Path("data/generated/segments")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scene_generator = SceneImageGenerator()
        
        # Character patterns for detection
        self.character_patterns = {
            'Emma': ['emma', 'her', 'she'],
            'Thomas': ['thomas', 'him', 'he', 'vendor', 'fruit vendor'],
            'James': ['james', 'him', 'he', 'brother'],
            'Father McKenzie': ['father mckenzie', 'priest', 'father', 'mckenzie'],
            'Mrs. Patterson': ['mrs. patterson', 'patterson', 'mrs'],
            'Mary': ['mary', 'her', 'she'],
            'John': ['john', 'him', 'he'],
            'Sarah': ['sarah', 'her', 'she']
        }
        
        logger.info("Initialized Chapter Segment Processor")
        logger.info("Will generate background settings and images for every 5-6 lines")
    
    def process_chapter(self, chapter_text: str, chapter_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chapter into segments with background generation"""
        
        chapter_id = chapter_info.get("chapter_number", 1)
        chapter_title = chapter_info.get("title", f"Chapter {chapter_id}")
        
        logger.info(f"Processing {chapter_title} into segments...")
        
        # Split chapter into segments (5-6 lines each)
        segments = self._split_into_segments(chapter_text)
        
        # Process each segment
        processed_segments = []
        
        for i, segment_text in enumerate(segments, 1):
            segment_info = self._analyze_segment(segment_text, i, chapter_id)
            
            # Generate scene image for segment
            scene_image = self.scene_generator.generate_scene_image(segment_info)
            segment_info["scene_image"] = scene_image
            
            processed_segments.append(segment_info)
            
            logger.info(f"✅ Processed segment {i}/{len(segments)}")
            
            # Brief pause to avoid overwhelming APIs
            time.sleep(2)
        
        # Create comprehensive chapter analysis
        chapter_analysis = {
            "chapter_info": chapter_info,
            "processing_summary": {
                "total_segments": len(processed_segments),
                "total_scenes_generated": sum(1 for s in processed_segments if s.get("scene_image")),
                "unique_characters": list(set().union(*[s["characters_present"] for s in processed_segments])),
                "unique_settings": list(set(s["detected_setting"] for s in processed_segments)),
                "processing_date": datetime.now().isoformat()
            },
            "segments": processed_segments
        }
        
        # Save chapter analysis
        output_file = self.output_dir / f"chapter_{chapter_id:02d}_segments.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chapter_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Chapter analysis saved: {output_file}")
        
        return chapter_analysis
    
    def _split_into_segments(self, text: str) -> List[str]:
        """Split text into segments of 5-6 lines"""
        
        # Split by lines and filter empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        segments = []
        current_segment = []
        
        for line in lines:
            current_segment.append(line)
            
            # Create segment every 5-6 lines
            if len(current_segment) >= 5:
                # Check if we should include one more line (max 6)
                if len(current_segment) == 5 and len([l for l in lines[lines.index(line)+1:] if l]) > 3:
                    # If there are more than 3 lines left, take one more for this segment
                    continue
                
                # Finalize current segment
                segments.append('\n'.join(current_segment))
                current_segment = []
        
        # Add remaining lines as final segment
        if current_segment:
            segments.append('\n'.join(current_segment))
        
        logger.info(f"Split into {len(segments)} segments")
        return segments
    
    def _analyze_segment(self, segment_text: str, segment_num: int, chapter_id: int) -> Dict[str, Any]:
        """Analyze a text segment for characters, setting, and context"""
        
        segment_id = f"{chapter_id:02d}_{segment_num:02d}"
        
        # Detect characters present in segment
        characters_present = self._detect_characters(segment_text)
        
        # Detect setting/location
        detected_setting = self._detect_setting(segment_text)
        
        # Analyze dialogue
        dialogue_info = self._analyze_dialogue(segment_text, characters_present)
        
        # Extract key phrases and actions
        key_actions = self._extract_key_actions(segment_text)
        
        # Determine scene mood and atmosphere
        scene_mood = self._analyze_scene_mood(segment_text)
        
        segment_info = {
            "segment_id": segment_id,
            "segment_number": segment_num,
            "chapter_id": chapter_id,
            "text": segment_text,
            "word_count": len(segment_text.split()),
            "line_count": len(segment_text.split('\n')),
            
            # Character information
            "characters_present": characters_present,
            "primary_character": self._identify_primary_character(segment_text, characters_present),
            "character_interactions": self._analyze_character_interactions(characters_present, dialogue_info),
            
            # Setting and scene information
            "detected_setting": detected_setting,
            "scene_mood": scene_mood,
            "time_context": self._detect_time_context(segment_text),
            "weather_context": self._detect_weather(segment_text),
            
            # Narrative elements
            "dialogue_info": dialogue_info,
            "key_actions": key_actions,
            "narrative_focus": self._determine_narrative_focus(segment_text),
            
            # Audio production hints
            "suggested_background_music": self._suggest_segment_music(scene_mood, detected_setting),
            "suggested_sound_effects": self._suggest_segment_sounds(segment_text, detected_setting),
            "narration_pace": self._suggest_narration_pace(segment_text, dialogue_info),
            
            # Image generation context
            "image_prompt_elements": {
                "characters": characters_present,
                "setting": detected_setting,
                "mood": scene_mood,
                "actions": key_actions,
                "atmosphere": self._describe_atmosphere(segment_text)
            }
        }
        
        return segment_info
    
    def _detect_characters(self, text: str) -> List[str]:
        """Detect which characters are present in the text segment"""
        
        text_lower = text.lower()
        characters_found = []
        
        for character, patterns in self.character_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    if character not in characters_found:
                        characters_found.append(character)
                    break
        
        # Sort by likely importance (based on name mentions vs pronouns)
        def character_importance(char):
            char_name = char.split()[0].lower()
            direct_mentions = text_lower.count(char_name)
            return direct_mentions
        
        characters_found.sort(key=character_importance, reverse=True)
        
        return characters_found
    
    def _detect_setting(self, text: str) -> str:
        """Detect the setting/location from text"""
        
        text_lower = text.lower()
        
        # Setting keywords mapping
        settings = {
            'marketplace': ['market', 'stall', 'vendor', 'crowd', 'buying', 'selling'],
            'cathedral': ['cathedral', 'church', 'priest', 'bell', 'prayer', 'altar'],
            'town square': ['square', 'fountain', 'town center', 'plaza'],
            'village': ['village', 'cottage', 'rural', 'countryside'],
            'forest': ['forest', 'tree', 'wood', 'path', 'nature'],
            'castle': ['castle', 'tower', 'wall', 'fortress', 'keep'],
            'london street': ['london', 'street', 'city', 'building'],
            'home': ['home', 'house', 'room', 'kitchen', 'door'],
            'bakery': ['baker', 'bread', 'bakery', 'oven'],
            'tavern': ['tavern', 'inn', 'ale', 'drink']
        }
        
        # Count matches for each setting
        setting_scores = {}
        for setting, keywords in settings.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                setting_scores[setting] = score
        
        if setting_scores:
            # Return setting with highest score
            best_setting = max(setting_scores, key=setting_scores.get)
            return best_setting
        
        # Default setting
        return "outdoor scene"
    
    def _analyze_dialogue(self, text: str, characters: List[str]) -> Dict[str, Any]:
        """Analyze dialogue in the segment"""
        
        # Find quoted text
        quotes = re.findall(r'"([^"]*)"', text)
        
        dialogue_info = {
            "has_dialogue": len(quotes) > 0,
            "quote_count": len(quotes),
            "quotes": quotes[:3],  # First 3 quotes
            "estimated_speakers": []
        }
        
        # Try to match quotes to characters
        if quotes and characters:
            lines = text.split('\n')
            for quote in quotes[:3]:
                # Look for character names near the quote
                speaker = self._identify_quote_speaker(quote, text, characters)
                dialogue_info["estimated_speakers"].append(speaker)
        
        return dialogue_info
    
    def _identify_quote_speaker(self, quote: str, full_text: str, characters: List[str]) -> str:
        """Try to identify who spoke a quote"""
        
        # Find the quote in context
        quote_pos = full_text.find(f'"{quote}"')
        if quote_pos == -1:
            return "Unknown"
        
        # Look at text around the quote
        context_start = max(0, quote_pos - 100)
        context_end = min(len(full_text), quote_pos + len(quote) + 100)
        context = full_text[context_start:context_end].lower()
        
        # Look for character names in context
        for character in characters:
            char_name = character.split()[0].lower()
            if char_name in context:
                return character
        
        return "Unknown"
    
    def _extract_key_actions(self, text: str) -> List[str]:
        """Extract key actions from the text"""
        
        text_lower = text.lower()
        actions = []
        
        # Action patterns
        action_patterns = {
            'walking': ['walked', 'walking', 'approached', 'moved', 'stepped'],
            'speaking': ['said', 'replied', 'called', 'whispered', 'shouted'],
            'looking': ['looked', 'watched', 'observed', 'saw', 'noticed'],
            'smiling': ['smiled', 'grinned', 'laughed'],
            'gesturing': ['waved', 'pointed', 'gestured', 'nodded'],
            'trading': ['bought', 'sold', 'paid', 'exchanged']
        }
        
        for action_type, patterns in action_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                actions.append(action_type)
        
        return actions[:3]  # Top 3 actions
    
    def _analyze_scene_mood(self, text: str) -> str:
        """Analyze the overall mood of the scene"""
        
        text_lower = text.lower()
        
        # Mood indicators
        if any(word in text_lower for word in ['smiled', 'laughed', 'cheerful', 'bright', 'happy']):
            return "cheerful"
        elif any(word in text_lower for word in ['worried', 'concerned', 'anxious', 'fear']):
            return "tense"
        elif any(word in text_lower for word in ['peaceful', 'calm', 'gentle', 'quiet']):
            return "peaceful"
        elif any(word in text_lower for word in ['busy', 'bustling', 'crowd', 'energy']):
            return "energetic"
        elif any(word in text_lower for word in ['dark', 'shadow', 'mysterious']):
            return "mysterious"
        else:
            return "neutral"
    
    def _identify_primary_character(self, text: str, characters: List[str]) -> str:
        """Identify the primary character in this segment"""
        
        if not characters:
            return "None"
        
        # Count direct name mentions for each character
        character_scores = {}
        text_lower = text.lower()
        
        for character in characters:
            char_name = character.split()[0].lower()
            score = text_lower.count(char_name)
            # Bonus for being mentioned early in the segment
            if char_name in text_lower[:100]:
                score += 1
            character_scores[character] = score
        
        if character_scores:
            return max(character_scores, key=character_scores.get)
        
        return characters[0] if characters else "None"
    
    def _analyze_character_interactions(self, characters: List[str], dialogue_info: Dict[str, Any]) -> List[str]:
        """Analyze interactions between characters"""
        
        interactions = []
        
        if len(characters) >= 2 and dialogue_info["has_dialogue"]:
            interactions.append("conversation")
        
        if len(characters) >= 2:
            interactions.append("character_meeting")
        
        if len(characters) == 1:
            interactions.append("solo_action")
        
        return interactions
    
    def _detect_time_context(self, text: str) -> str:
        """Detect time context from text"""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['morning', 'dawn']):
            return "morning"
        elif any(word in text_lower for word in ['noon', 'midday']):
            return "midday"
        elif any(word in text_lower for word in ['evening', 'sunset']):
            return "evening"
        elif any(word in text_lower for word in ['night', 'dark']):
            return "night"
        else:
            return "unspecified"
    
    def _detect_weather(self, text: str) -> str:
        """Detect weather context"""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['sun', 'sunny', 'bright']):
            return "sunny"
        elif any(word in text_lower for word in ['rain', 'wet', 'storm']):
            return "rainy"
        elif any(word in text_lower for word in ['wind', 'windy', 'breeze']):
            return "windy"
        elif any(word in text_lower for word in ['cloud', 'overcast']):
            return "cloudy"
        else:
            return "clear"
    
    def _determine_narrative_focus(self, text: str) -> str:
        """Determine what the narrative is focusing on"""
        
        text_lower = text.lower()
        
        if '"' in text:
            return "dialogue"
        elif any(word in text_lower for word in ['walked', 'moved', 'approached']):
            return "action"
        elif any(word in text_lower for word in ['looked', 'saw', 'noticed']):
            return "observation"
        elif any(word in text_lower for word in ['thought', 'wondered', 'realized']):
            return "internal"
        else:
            return "description"
    
    def _suggest_segment_music(self, mood: str, setting: str) -> str:
        """Suggest background music for segment"""
        
        music_map = {
            ("cheerful", "marketplace"): "lively medieval town music",
            ("peaceful", "cathedral"): "sacred, gentle choral",
            ("tense", "any"): "subtle tension music",
            ("energetic", "any"): "upbeat background music",
            ("mysterious", "any"): "mysterious, atmospheric"
        }
        
        # Try specific combinations first
        for (mood_key, setting_key), music in music_map.items():
            if mood == mood_key and (setting_key == "any" or setting_key in setting):
                return music
        
        # Default based on setting
        setting_music = {
            "marketplace": "medieval town ambience",
            "cathedral": "sacred music",
            "forest": "nature, mystical",
            "castle": "grand, orchestral"
        }
        
        return setting_music.get(setting, "gentle atmospheric music")
    
    def _suggest_segment_sounds(self, text: str, setting: str) -> List[str]:
        """Suggest sound effects for segment"""
        
        text_lower = text.lower()
        sounds = []
        
        # Text-based sounds
        if 'footsteps' in text_lower or 'walked' in text_lower:
            sounds.append("footsteps")
        if 'bell' in text_lower:
            sounds.append("church bells")
        if 'crowd' in text_lower or 'market' in text_lower:
            sounds.append("crowd chatter")
        if 'wind' in text_lower:
            sounds.append("wind ambience")
        
        # Setting-based sounds
        setting_sounds = {
            "marketplace": ["crowd chatter", "vendor calls"],
            "cathedral": ["church bells", "echo"],
            "forest": ["birds", "leaves rustling"],
            "village": ["peaceful ambience"]
        }
        
        if setting in setting_sounds:
            sounds.extend(setting_sounds[setting])
        
        return list(set(sounds))  # Remove duplicates
    
    def _suggest_narration_pace(self, text: str, dialogue_info: Dict[str, Any]) -> str:
        """Suggest narration pace for segment"""
        
        if dialogue_info["quote_count"] > 2:
            return "varied pace for dialogue"
        elif len(text.split()) > 100:
            return "measured, steady pace"
        elif any(word in text.lower() for word in ['quickly', 'rushed', 'hurried']):
            return "faster pace"
        elif any(word in text.lower() for word in ['slowly', 'gentle', 'calm']):
            return "slower pace"
        else:
            return "normal pace"
    
    def _describe_atmosphere(self, text: str) -> str:
        """Describe the atmospheric context for image generation"""
        
        text_lower = text.lower()
        
        atmosphere_elements = []
        
        if any(word in text_lower for word in ['bright', 'sun', 'light']):
            atmosphere_elements.append("bright lighting")
        elif any(word in text_lower for word in ['dark', 'shadow']):
            atmosphere_elements.append("moody lighting")
        
        if any(word in text_lower for word in ['busy', 'crowd', 'bustling']):
            atmosphere_elements.append("busy atmosphere")
        elif any(word in text_lower for word in ['quiet', 'peaceful', 'calm']):
            atmosphere_elements.append("peaceful atmosphere")
        
        return ", ".join(atmosphere_elements) if atmosphere_elements else "natural atmosphere"
    
    def generate_chapter_report(self, chapter_analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive report for the chapter"""
        
        chapter_id = chapter_analysis["chapter_info"]["chapter_number"]
        segments = chapter_analysis["segments"]
        summary = chapter_analysis["processing_summary"]
        
        report_lines = [
            f"Chapter {chapter_id} Segment Analysis Report",
            "=" * 50,
            "",
            f"Total Segments: {summary['total_segments']}",
            f"Scene Images Generated: {summary['total_scenes_generated']}",
            f"Characters Featured: {', '.join(summary['unique_characters'])}",
            f"Settings Detected: {', '.join(summary['unique_settings'])}",
            "",
            "Segment Details:",
            "-" * 30
        ]
        
        for segment in segments:
            segment_lines = [
                f"Segment {segment['segment_number']}:",
                f"  Characters: {', '.join(segment['characters_present']) if segment['characters_present'] else 'None'}",
                f"  Setting: {segment['detected_setting']}",
                f"  Mood: {segment['scene_mood']}",
                f"  Dialogue: {'Yes' if segment['dialogue_info']['has_dialogue'] else 'No'}",
                f"  Scene Image: {Path(segment['scene_image']).name if segment.get('scene_image') else 'None'}",
                ""
            ]
            report_lines.extend(segment_lines)
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / f"chapter_{chapter_id:02d}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✅ Chapter report saved: {report_file}")
        
        return str(report_file)


def main():
    """Main function to demonstrate chapter segment processing"""
    
    print("EchoTales Chapter Segment Processor")
    print("=" * 50)
    print("🖼️  Scene images generated for every 5-6 lines")
    print("👥 Character context provided for each segment")
    print("🎭 Background settings and atmosphere analysis")
    print()
    
    # Sample chapter text
    sample_chapter = '''Emma walked through the busy marketplace in London, her basket clutched tightly in her hand. The morning sun cast long shadows between the wooden stalls.

"Fresh apples! Best in all of England!" called out Thomas, the fruit vendor, waving to catch her attention.

Emma smiled and approached his stall. "Good morning, Thomas. How much for a dozen?"

"For you, my dear Emma, just two shillings," Thomas replied with a grin. "Your father was always good to me."

"Thank you," Emma said, placing the coins in his weathered palm. "Has anyone seen my brother James today? He was supposed to meet me here."

Thomas shook his head. "No, but Mrs. Patterson mentioned she saw him near the cathedral this morning."

Emma nodded and continued through the market. The smell of fresh bread from Baker Street filled her nostrils. She needed to find James before noon - their mother would be expecting them home for lunch.

Near the fountain in the town square, she spotted a familiar figure. "James!" she called out.

Her brother turned around, his face brightening. "Emma! I've been looking everywhere for you. Wait until you hear what happened at the cathedral."

"What happened?" Emma asked, setting down her basket.

"Father McKenzie told me the most extraordinary story about our grandfather's service in the war," James said, his eyes wide with excitement. "Apparently, he saved an entire village in France!"

Emma gasped. "Our grandfather? But mother never told us about that!"

"That's because she doesn't know," James whispered. "Father McKenzie made me promise to keep it secret until the right time."

The two siblings walked home together, both lost in thought about their family's hidden history.'''
    
    # Sample chapter info
    chapter_info = {
        "chapter_number": 1,
        "title": "Chapter 1: The Marketplace Discovery",
        "page_number": 1,
        "word_count": len(sample_chapter.split())
    }
    
    # Initialize processor
    processor = ChapterSegmentProcessor()
    
    print("Processing sample chapter...")
    print(f"Chapter: {chapter_info['title']}")
    print(f"Word count: {chapter_info['word_count']}")
    print()
    
    # Process the chapter
    result = processor.process_chapter(sample_chapter, chapter_info)
    
    if result:
        print("✅ Chapter processing complete!")
        print(f"📊 Summary:")
        
        summary = result["processing_summary"]
        print(f"  📄 Total segments: {summary['total_segments']}")
        print(f"  🖼️  Scene images: {summary['total_scenes_generated']}")
        print(f"  👥 Characters found: {', '.join(summary['unique_characters'])}")
        print(f"  🏞️  Settings detected: {', '.join(summary['unique_settings'])}")
        
        # Generate and display report
        report_file = processor.generate_chapter_report(result)
        print(f"\n📋 Detailed report: {report_file}")
        
        # Show sample segment info
        if result["segments"]:
            sample_segment = result["segments"][0]
            print(f"\n🎬 Sample Segment Analysis:")
            print(f"  ID: {sample_segment['segment_id']}")
            print(f"  Characters: {', '.join(sample_segment['characters_present'])}")
            print(f"  Setting: {sample_segment['detected_setting']}")
            print(f"  Mood: {sample_segment['scene_mood']}")
            print(f"  Scene image: {Path(sample_segment['scene_image']).name}")
        
        print(f"\n📂 Check data/generated/segments/ for all outputs!")
    else:
        print("❌ Chapter processing failed")


if __name__ == "__main__":
    main()