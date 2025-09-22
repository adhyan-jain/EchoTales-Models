from flask import Flask, request, jsonify, send_file
from flask_restful import Api, Resource
from flask_cors import CORS
import redis
import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path

from ..core.models import (
    ProcessedNovel, Character, Scene, BackgroundSetting, 
    DialogueLine, VoiceProfile, PersonalityVector, PhysicalDescription
)
from ..processing.enhanced_character_processor import EnhancedCharacterProcessor
from ..processing.scene_analyzer import SceneAnalyzer
from ..voice.classifier import VoiceDatasetClassifier
from ..ai.image_generator import ImageGenerationService, ImageRequest, create_image_service
from ..utils.config_loader import ConfigLoader
from ..utils.logger_setup import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for NextJS integration
api = Api(app)

# Global variables for system components
config = None
redis_client = None
character_processor = None
scene_analyzer = None
voice_classifier = None
image_service = None
executor = ThreadPoolExecutor(max_workers=4)


def initialize_components():
    """Initialize all system components"""
    global config, redis_client, character_processor, scene_analyzer, voice_classifier, image_service
    
    try:
        # Load configuration
        config = ConfigLoader()
        logger.info("Configuration loaded successfully")
        
        # Initialize Redis client for caching
        try:
            redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                decode_responses=config.redis.decode_responses
            )
            redis_client.ping()  # Test connection
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            redis_client = None
        
        # Initialize processing components
        character_processor = EnhancedCharacterProcessor()
        scene_analyzer = SceneAnalyzer()
        
        # Initialize voice classifier
        voice_classifier = VoiceDatasetClassifier(
            dataset_path=config.voice_dataset.path,
            models_path="models/voice_cloning/",
            gemini_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Initialize image generation service
        output_dir = config.get_config_value('image_generation.output_directory', 'data/output/images')
        image_service = ImageGenerationService(output_dir=output_dir)
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        raise


class HealthCheckResource(Resource):
    """Health check endpoint"""
    
    def get(self):
        """Check system health"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'character_processor': character_processor is not None,
                'scene_analyzer': scene_analyzer is not None,
                'voice_classifier': voice_classifier is not None,
                'image_service': image_service is not None,
                'redis': redis_client is not None and self._test_redis()
            }
        }
        
        # Overall status
        all_healthy = all(health_status['components'].values())
        health_status['status'] = 'healthy' if all_healthy else 'degraded'
        
        return health_status, 200 if all_healthy else 503
    
    def _test_redis(self) -> bool:
        """Test Redis connection"""
        try:
            if redis_client:
                redis_client.ping()
                return True
        except:
            pass
        return False


class NovelProcessingResource(Resource):
    """Process novels and extract structured data"""
    
    def post(self):
        """Process a novel and return structured data"""
        try:
            # Parse request data
            data = request.get_json()
            if not data:
                return {'error': 'No JSON data provided'}, 400
            
            # Validate required fields
            required_fields = ['book_id', 'title', 'content']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return {'error': f'Missing required fields: {missing_fields}'}, 400
            
            book_id = data['book_id']
            title = data['title']
            content = data['content']
            author = data.get('author')
            
            # Check cache first
            cached_result = self._get_cached_result(book_id)
            if cached_result:
                logger.info(f"Returning cached result for book {book_id}")
                return cached_result, 200
            
            # Process novel
            logger.info(f"Starting novel processing for: {title}")
            
            # Split content into chapters
            chapters = self._split_into_chapters(content)
            
            # Extract characters using BookNLP (placeholder - would integrate actual BookNLP)
            raw_characters = self._extract_characters_booknlp(content)
            
            # Extract dialogue
            dialogue_lines = self._extract_dialogue(chapters)
            
            # Enhance character analysis
            enhanced_characters = self._enhance_characters(raw_characters, dialogue_lines)
            
            # Analyze scenes and settings
            settings, transitions = scene_analyzer.extract_settings_and_transitions(chapters)
            scenes = self._create_scenes(chapters, enhanced_characters, dialogue_lines, settings)
            
            # Match voices to characters
            self._match_voices_to_characters(enhanced_characters)
            
            # Create processed novel object
            processed_novel = ProcessedNovel(
                book_id=book_id,
                title=title,
                author=author,
                characters=enhanced_characters,
                scenes=scenes,
                settings=settings,
                setting_transitions=transitions,
                total_chapters=len(chapters),
                processing_timestamp=datetime.utcnow(),
                metadata={
                    'processing_version': '2.0.0',
                    'total_dialogue_lines': len(dialogue_lines),
                    'total_settings': len(settings)
                }
            )
            
            # Convert to dict for JSON response
            result = processed_novel.dict()
            
            # Cache the result
            self._cache_result(book_id, result)
            
            logger.info(f"Novel processing completed for: {title}")
            return result, 200
            
        except Exception as e:
            logger.error(f"Novel processing failed: {e}")
            return {'error': f'Processing failed: {str(e)}'}, 500
    
    def _get_cached_result(self, book_id: str) -> Optional[Dict]:
        """Get cached processing result"""
        if not redis_client:
            return None
        
        try:
            cache_key = f"novel:{book_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_result(self, book_id: str, result: Dict):
        """Cache processing result"""
        if not redis_client:
            return
        
        try:
            cache_key = f"novel:{book_id}"
            ttl = config.caching.ttl if config else 3600
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)  # default=str for datetime serialization
            )
            logger.info(f"Result cached for book {book_id}")
        except Exception as e:
            logger.warning(f"Caching failed: {e}")
    
    def _split_into_chapters(self, content: str) -> List[str]:
        """Split content into chapters"""
        # Simple chapter detection - can be enhanced
        chapter_patterns = [
            r'\n\s*Chapter\s+\d+',
            r'\n\s*CHAPTER\s+\d+',
            r'\n\s*\d+\.',
            r'\n\s*[IVX]+\.',
        ]
        
        import re
        chapters = []
        current_chapter = []
        
        lines = content.split('\n')
        
        for line in lines:
            # Check if line indicates new chapter
            is_new_chapter = any(re.match(pattern, '\n' + line) for pattern in chapter_patterns)
            
            if is_new_chapter and current_chapter:
                # Save current chapter and start new one
                chapters.append('\n'.join(current_chapter))
                current_chapter = [line]
            else:
                current_chapter.append(line)
        
        # Add final chapter
        if current_chapter:
            chapters.append('\n'.join(current_chapter))
        
        # If no chapters detected, treat entire content as one chapter
        if len(chapters) <= 1:
            chapters = [content]
        
        return chapters
    
    def _extract_characters_booknlp(self, content: str) -> List[Dict]:
        """Extract characters using BookNLP (placeholder implementation)"""
        # This would integrate with actual BookNLP library
        # For now, return mock character data
        
        import re
        
        # Simple name extraction (placeholder)
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        name_counts = {}
        
        for name in potential_names:
            if len(name.split()) <= 2 and len(name) > 2:  # Simple filtering
                name_counts[name] = name_counts.get(name, 0) + 1
        
        # Get most frequent names as main characters
        sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
        main_characters = sorted_names[:10]  # Top 10 most frequent
        
        characters = []
        for i, (name, count) in enumerate(main_characters):
            characters.append({
                'booknlp_id': i,
                'name': name,
                'mention_count': count,
                'context_sentences': self._find_character_context(content, name)
            })
        
        return characters
    
    def _find_character_context(self, content: str, character_name: str) -> List[str]:
        """Find context sentences mentioning the character"""
        import re
        
        sentences = re.split(r'[.!?]+', content)
        context_sentences = []
        
        for sentence in sentences:
            if character_name in sentence:
                context_sentences.append(sentence.strip())
                if len(context_sentences) >= 5:  # Limit context
                    break
        
        return context_sentences
    
    def _extract_dialogue(self, chapters: List[str]) -> List[DialogueLine]:
        """Extract dialogue from chapters"""
        import re
        
        dialogue_lines = []
        line_id_counter = 0
        
        for chapter_idx, chapter in enumerate(chapters):
            paragraphs = chapter.split('\n\n')
            
            for para_idx, paragraph in enumerate(paragraphs):
                # Find quoted dialogue
                dialogue_pattern = r'"([^"]+)"'
                quotes = re.findall(dialogue_pattern, paragraph)
                
                for quote in quotes:
                    if len(quote.strip()) > 5:  # Filter out short quotes
                        dialogue_lines.append(DialogueLine(
                            line_id=f"line_{line_id_counter}",
                            character_id="unknown",  # Will be matched later
                            text=quote.strip(),
                            emotion="neutral",  # Will be analyzed later
                            chapter=chapter_idx + 1,
                            paragraph=para_idx,
                            setting_id=None
                        ))
                        line_id_counter += 1
        
        return dialogue_lines
    
    def _enhance_characters(self, raw_characters: List[Dict], dialogue_lines: List[DialogueLine]) -> List[Character]:
        """Enhance character analysis using ML models"""
        enhanced_characters = []
        
        for char_data in raw_characters:
            try:
                # Extract personality
                personality = character_processor.extract_enhanced_personality(
                    char_data.get('context_sentences', []),
                    [d.text for d in dialogue_lines if char_data['name'].lower() in d.text.lower()]
                )
                
                # Extract physical description
                physical_desc = character_processor.extract_physical_description(
                    char_data.get('context_sentences', [])
                )
                
                # Determine gender and age (simplified)
                gender = self._infer_gender(char_data['name'])
                age_group = self._infer_age_group(char_data.get('context_sentences', []))
                
                # Create enhanced character
                character = Character(
                    character_id=f"char_{char_data['booknlp_id']}",
                    name=char_data['name'],
                    booknlp_id=char_data['booknlp_id'],
                    gender=gender,
                    age_group=age_group,
                    personality=personality,
                    physical_description=physical_desc,
                    importance_score=min(char_data.get('mention_count', 1) / 100.0, 1.0),
                    first_appearance=1,  # Simplified
                    last_appearance=1,   # Simplified
                    total_dialogue_lines=len([d for d in dialogue_lines if char_data['name'].lower() in d.text.lower()])
                )
                
                enhanced_characters.append(character)
                
            except Exception as e:
                logger.warning(f"Character enhancement failed for {char_data['name']}: {e}")
                continue
        
        return enhanced_characters
    
    def _infer_gender(self, name: str) -> str:
        """Simple gender inference from name"""
        # This is a simplified approach - in production, use a proper name database
        male_names = ['john', 'james', 'robert', 'michael', 'william', 'david', 'richard']
        female_names = ['mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara', 'susan']
        
        first_name = name.split()[0].lower()
        
        if first_name in male_names:
            return 'male'
        elif first_name in female_names:
            return 'female'
        else:
            return 'unknown'
    
    def _infer_age_group(self, context_sentences: List[str]) -> str:
        """Infer age group from context"""
        context_text = ' '.join(context_sentences).lower()
        
        if any(word in context_text for word in ['child', 'kid', 'young', 'boy', 'girl']):
            return 'child'
        elif any(word in context_text for word in ['teen', 'teenager', 'adolescent']):
            return 'teenager'
        elif any(word in context_text for word in ['elderly', 'old', 'aged', 'senior']):
            return 'elderly'
        else:
            return 'adult'
    
    def _create_scenes(self, chapters: List[str], characters: List[Character], 
                      dialogue_lines: List[DialogueLine], settings: List[BackgroundSetting]) -> List[Scene]:
        """Create scene objects from processed data"""
        scenes = []
        scene_id_counter = 0
        
        for chapter_idx, chapter in enumerate(chapters):
            paragraphs = chapter.split('\n\n')
            
            # Simple scene detection - group paragraphs into scenes
            current_scene_paras = []
            
            for para_idx, paragraph in enumerate(paragraphs):
                current_scene_paras.append(paragraph)
                
                # End scene if we hit a setting change or chapter end
                if (para_idx == len(paragraphs) - 1 or 
                    len(current_scene_paras) >= 5):  # Max 5 paragraphs per scene
                    
                    # Find relevant setting
                    scene_setting = settings[0] if settings else BackgroundSetting(
                        setting_id="default_setting",
                        name="Unknown Location",
                        type="unknown",
                        description="Default setting",
                        atmosphere="neutral",
                        realm="real_world",
                        image_prompt="generic location"
                    )
                    
                    # Find characters present
                    scene_text = ' '.join(current_scene_paras)
                    chars_present = [
                        char.character_id for char in characters
                        if char.name.lower() in scene_text.lower()
                    ]
                    
                    # Find dialogue in this scene
                    scene_dialogue = [
                        d for d in dialogue_lines
                        if d.chapter == chapter_idx + 1 and 
                        para_idx - len(current_scene_paras) <= d.paragraph <= para_idx
                    ]
                    
                    scene = Scene(
                        scene_id=f"scene_{scene_id_counter}",
                        chapter=chapter_idx + 1,
                        start_paragraph=max(0, para_idx - len(current_scene_paras) + 1),
                        end_paragraph=para_idx,
                        setting=scene_setting,
                        characters_present=chars_present,
                        dialogue_lines=scene_dialogue,
                        narrative_summary=scene_text[:200],  # First 200 chars
                        mood="neutral"  # Simplified
                    )
                    
                    scenes.append(scene)
                    scene_id_counter += 1
                    current_scene_paras = []
        
        return scenes
    
    def _match_voices_to_characters(self, characters: List[Character]):
        """Match voice profiles to characters"""
        if not voice_classifier:
            return
        
        try:
            # Get available voice profiles
            voice_profiles = voice_classifier.process_actor_dataset()
            all_profiles = voice_profiles.get('male', []) + voice_profiles.get('female', [])
            
            for character in characters:
                # Match character to best voice
                matched_voice = voice_classifier.match_character_to_voice(character, all_profiles)
                if matched_voice:
                    character.voice_profile = matched_voice
                    
        except Exception as e:
            logger.warning(f"Voice matching failed: {e}")


class CharacterResource(Resource):
    """Character management endpoints"""
    
    def get(self, book_id, character_id=None):
        """Get character(s) for a book"""
        try:
            if character_id:
                # Get specific character
                character_data = self._get_character(book_id, character_id)
                if not character_data:
                    return {'error': 'Character not found'}, 404
                return character_data, 200
            else:
                # Get all characters for book
                characters = self._get_all_characters(book_id)
                return {'characters': characters}, 200
                
        except Exception as e:
            logger.error(f"Character retrieval failed: {e}")
            return {'error': 'Failed to retrieve character data'}, 500
    
    def _get_character(self, book_id: str, character_id: str) -> Optional[Dict]:
        """Get specific character data"""
        # In production, this would query your NextJS backend/MongoDB
        # For now, check cache
        cached_novel = self._get_cached_novel(book_id)
        if cached_novel:
            for char in cached_novel.get('characters', []):
                if char['character_id'] == character_id:
                    return char
        return None
    
    def _get_all_characters(self, book_id: str) -> List[Dict]:
        """Get all characters for a book"""
        cached_novel = self._get_cached_novel(book_id)
        if cached_novel:
            return cached_novel.get('characters', [])
        return []
    
    def _get_cached_novel(self, book_id: str) -> Optional[Dict]:
        """Get cached novel data"""
        if not redis_client:
            return None
        
        try:
            cache_key = f"novel:{book_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None


class VoiceGenerationResource(Resource):
    """Voice generation endpoints"""
    
    def post(self):
        """Generate audio for specific characters/chapters"""
        try:
            data = request.get_json()
            if not data:
                return {'error': 'No JSON data provided'}, 400
            
            # Validate required fields
            required_fields = ['book_id', 'text', 'character_id']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return {'error': f'Missing required fields: {missing_fields}'}, 400
            
            book_id = data['book_id']
            text = data['text']
            character_id = data['character_id']
            tts_engine = data.get('tts_engine', 'openvoice')
            
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Submit job to background processor
            future = executor.submit(
                self._generate_audio_background,
                job_id, book_id, character_id, text, tts_engine
            )
            
            return {
                'job_id': job_id,
                'status': 'processing',
                'message': 'Audio generation started'
            }, 202
            
        except Exception as e:
            logger.error(f"Voice generation request failed: {e}")
            return {'error': 'Failed to start audio generation'}, 500
    
    def get(self, job_id):
        """Check audio generation status"""
        try:
            # Check job status in cache/database
            status_data = self._get_job_status(job_id)
            if not status_data:
                return {'error': 'Job not found'}, 404
            
            return status_data, 200
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {'error': 'Failed to check job status'}, 500
    
    def _generate_audio_background(self, job_id: str, book_id: str, 
                                 character_id: str, text: str, tts_engine: str):
        """Background audio generation task"""
        try:
            # Update job status
            self._update_job_status(job_id, 'processing', 'Generating audio...')
            
            # Placeholder for actual TTS generation
            # In production, this would use the selected TTS engine
            audio_file_path = self._generate_tts_audio(text, character_id, tts_engine)
            
            # Update job status with result
            self._update_job_status(job_id, 'completed', 'Audio generated successfully', {
                'audio_file_path': audio_file_path,
                'duration': 10.0  # Placeholder duration
            })
            
        except Exception as e:
            logger.error(f"Audio generation failed for job {job_id}: {e}")
            self._update_job_status(job_id, 'failed', f'Generation failed: {str(e)}')
    
    def _generate_tts_audio(self, text: str, character_id: str, tts_engine: str) -> str:
        """Generate TTS audio (placeholder implementation)"""
        # In production, this would integrate with actual TTS engines
        # For now, return a placeholder path
        
        output_dir = Path("data/output/audio/characters")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_filename = f"{character_id}_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = output_dir / audio_filename
        
        # Placeholder: create empty file
        audio_path.touch()
        
        return str(audio_path)
    
    def _get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status from cache"""
        if not redis_client:
            return None
        
        try:
            cache_key = f"audio_job:{job_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Job status retrieval failed: {e}")
        
        return None
    
    def _update_job_status(self, job_id: str, status: str, message: str, result: Optional[Dict] = None):
        """Update job status in cache"""
        if not redis_client:
            return
        
        try:
            cache_key = f"audio_job:{job_id}"
            status_data = {
                'job_id': job_id,
                'status': status,
                'message': message,
                'timestamp': datetime.utcnow().isoformat(),
                'result': result
            }
            
            # Cache for 24 hours
            redis_client.setex(cache_key, 86400, json.dumps(status_data))
            
        except Exception as e:
            logger.warning(f"Job status update failed: {e}")


class ImageGenerationResource(Resource):
    """Image generation endpoints"""
    
    def post(self):
        """Generate character portraits and scene images"""
        try:
            data = request.get_json()
            if not data:
                return {'error': 'No JSON data provided'}, 400
            
            # Validate required fields
            required_fields = ['type', 'prompt']  # type: 'character' or 'scene'
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return {'error': f'Missing required fields: {missing_fields}'}, 400
            
            image_type = data['type']
            prompt = data['prompt']
            book_id = data.get('book_id')
            style = data.get('style', 'cinematic')
            size = data.get('size', '1024x1024')
            quality = data.get('quality', 'standard')
            
            # Validate image type
            if image_type not in ['character', 'scene', 'background']:
                return {'error': 'Invalid image type. Must be: character, scene, or background'}, 400
            
            # Validate style
            available_styles = ['cinematic', 'fantasy', 'realistic', 'artistic', 'anime', 'comic']
            if style not in available_styles:
                return {'error': f'Invalid style. Available styles: {available_styles}'}, 400
            
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Submit job to background processor
            future = executor.submit(
                self._generate_image_background,
                job_id, image_type, prompt, style, book_id, size, quality
            )
            
            return {
                'job_id': job_id,
                'status': 'processing',
                'message': 'Image generation started',
                'estimated_time': '30-60 seconds'
            }, 202
            
        except Exception as e:
            logger.error(f"Image generation request failed: {e}")
            return {'error': 'Failed to start image generation'}, 500
    
    def get(self, job_id):
        """Check image generation status"""
        try:
            # Check job status in cache
            status_data = self._get_image_job_status(job_id)
            if not status_data:
                return {'error': 'Job not found'}, 404
            
            return status_data, 200
            
        except Exception as e:
            logger.error(f"Image status check failed: {e}")
            return {'error': 'Failed to check job status'}, 500
    
    def _generate_image_background(self, job_id: str, image_type: str, 
                                 prompt: str, style: str, book_id: Optional[str],
                                 size: str = "1024x1024", quality: str = "standard"):
        """Background image generation task using actual image generation service"""
        try:
            # Update job status
            self._update_image_job_status(job_id, 'processing', 'Generating image...')
            
            # Use actual image generation service
            if image_service is None:
                raise Exception("Image generation service not available")
            
            # Generate image using the service
            result = image_service.generate_image(
                prompt=prompt,
                style=style,
                image_type=image_type,
                size=size,
                quality=quality
            )
            
            # Update job status with result
            self._update_image_job_status(job_id, 'completed', 'Image generated successfully', {
                'image_file_path': result['local_path'],
                'image_url': f'/api/images/{job_id}',  # URL to serve the image
                'prompt': result['enhanced_prompt'],
                'style': style,
                'provider': result['provider_used'],
                'generation_time': result['generation_time']
            })
            
        except Exception as e:
            logger.error(f"Image generation failed for job {job_id}: {e}")
            self._update_image_job_status(job_id, 'failed', f'Generation failed: {str(e)}')
    
    def _get_image_job_status(self, job_id: str) -> Optional[Dict]:
        """Get image job status from cache"""
        if not redis_client:
            return None
        
        try:
            cache_key = f"image_job:{job_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Image job status retrieval failed: {e}")
        
        return None
    
    def _update_image_job_status(self, job_id: str, status: str, message: str, result: Optional[Dict] = None):
        """Update image job status in cache"""
        if not redis_client:
            return
        
        try:
            cache_key = f"image_job:{job_id}"
            status_data = {
                'job_id': job_id,
                'status': status,
                'message': message,
                'timestamp': datetime.utcnow().isoformat(),
                'result': result
            }
            
            # Cache for 24 hours
            redis_client.setex(cache_key, 86400, json.dumps(status_data))
            
        except Exception as e:
            logger.warning(f"Image job status update failed: {e}")
    
    def generate_character_portrait(self, character_data: Dict, style: str = "cinematic") -> str:
        """Generate a character portrait from character data"""
        try:
            # Build prompt from character data
            prompt = self._build_character_prompt(character_data)
            
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Submit generation job
            future = executor.submit(
                self._generate_image_background,
                job_id, "character", prompt, style, character_data.get('book_id')
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Character portrait generation failed: {e}")
            raise
    
    def generate_scene_image(self, scene_data: Dict, style: str = "cinematic") -> str:
        """Generate a scene image from scene data"""
        try:
            # Build prompt from scene data
            prompt = self._build_scene_prompt(scene_data)
            
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Submit generation job
            future = executor.submit(
                self._generate_image_background,
                job_id, "scene", prompt, style, scene_data.get('book_id')
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Scene image generation failed: {e}")
            raise
    
    def _build_character_prompt(self, character_data: Dict) -> str:
        """Build an image generation prompt from character data"""
        name = character_data.get('name', 'Unknown Character')
        gender = character_data.get('gender', 'person')
        age_group = character_data.get('age_group', 'adult')
        personality = character_data.get('personality', {})
        physical_desc = character_data.get('physical_description', {})
        
        # Start with basic description
        prompt_parts = [f"Portrait of {name}, {age_group} {gender}"]
        
        # Add physical description
        if physical_desc:
            if physical_desc.get('hair_color'):
                prompt_parts.append(f"with {physical_desc['hair_color']} hair")
            if physical_desc.get('eye_color'):
                prompt_parts.append(f"and {physical_desc['eye_color']} eyes")
            if physical_desc.get('build'):
                prompt_parts.append(f"{physical_desc['build']} build")
        
        # Add personality traits to expression
        if personality:
            traits = personality.get('traits', [])
            if 'confident' in traits:
                prompt_parts.append("confident expression")
            elif 'mysterious' in traits:
                prompt_parts.append("mysterious expression")
            elif 'kind' in traits:
                prompt_parts.append("warm, kind expression")
        
        prompt_parts.extend([
            "high quality portrait",
            "detailed",
            "professional lighting",
            "8k resolution"
        ])
        
        return ", ".join(prompt_parts)
    
    def _build_scene_prompt(self, scene_data: Dict) -> str:
        """Build an image generation prompt from scene data"""
        setting = scene_data.get('setting', {})
        mood = scene_data.get('mood', 'neutral')
        characters = scene_data.get('characters_present', [])
        narrative = scene_data.get('narrative_summary', '')
        
        # Start with setting description
        prompt_parts = []
        
        if setting:
            setting_type = setting.get('type', 'location')
            setting_name = setting.get('name', 'unknown location')
            atmosphere = setting.get('atmosphere', 'neutral')
            
            prompt_parts.append(f"{setting_name}, {setting_type}")
            if atmosphere != 'neutral':
                prompt_parts.append(f"{atmosphere} atmosphere")
        
        # Add character count if specified
        if characters:
            char_count = len(characters)
            if char_count == 1:
                prompt_parts.append("single character")
            elif char_count == 2:
                prompt_parts.append("two characters")
            elif char_count > 2:
                prompt_parts.append(f"{char_count} characters")
        
        # Add mood
        if mood != 'neutral':
            prompt_parts.append(f"{mood} mood")
        
        # Add narrative context if available
        if narrative:
            # Extract key visual elements from narrative
            narrative_lower = narrative.lower()
            if 'night' in narrative_lower or 'dark' in narrative_lower:
                prompt_parts.append("night scene")
            elif 'day' in narrative_lower or 'bright' in narrative_lower:
                prompt_parts.append("daylight scene")
        
        prompt_parts.extend([
            "cinematic composition",
            "detailed environment",
            "atmospheric lighting",
            "8k resolution"
        ])
        
        return ", ".join(prompt_parts)


# Register API routes
api.add_resource(HealthCheckResource, '/api/health')
api.add_resource(NovelProcessingResource, '/api/novels/process')
api.add_resource(CharacterResource, 
                '/api/novels/<string:book_id>/characters',
                '/api/novels/<string:book_id>/characters/<string:character_id>')
api.add_resource(VoiceGenerationResource, 
                '/api/audio/generate',
                '/api/audio/status/<string:job_id>')
api.add_resource(ImageGenerationResource,
                '/api/images/generate', 
                '/api/images/status/<string:job_id>')


@app.route('/api/images/<string:job_id>')
def serve_generated_image(job_id):
    """Serve generated images"""
    try:
        # Get job status to find image path
        if redis_client:
            cache_key = f"image_job:{job_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                job_data = json.loads(cached_data)
                if job_data['status'] == 'completed' and 'result' in job_data:
                    image_path = job_data['result']['image_file_path']
                    if os.path.exists(image_path):
                        return send_file(image_path)
        
        return {'error': 'Image not found'}, 404
        
    except Exception as e:
        logger.error(f"Image serving failed: {e}")
        return {'error': 'Failed to serve image'}, 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


def create_app():
    """Application factory"""
    initialize_components()
    return app


if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    # Run the app
    host = config.app.host if config else '127.0.0.1'
    port = config.app.port if config else 5000
    debug = config.app.debug if config else True
    
    logger.info(f"Starting EchoTales API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)