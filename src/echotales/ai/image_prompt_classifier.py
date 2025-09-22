"""
Image Prompt Classifier for EchoTales

This module optimizes image generation prompts for different AI models and styles.
It analyzes character descriptions and scenes to create optimized prompts that work
best with specific image generation APIs (Gemini, OpenAI, HuggingFace, etc.).

Features:
- Style-specific prompt optimization
- Model-specific prompt adaptation
- Character portrait prompt enhancement
- Scene visualization prompt generation
- Negative prompt suggestions
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# ML Libraries for text analysis
try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..core.models import Character, PhysicalDescription, PersonalityVector, Scene
from ..ai.model_manager import ModelManager, create_model_manager

logger = logging.getLogger(__name__)


class ImageStyle(Enum):
    """Supported image styles"""
    PHOTOREALISTIC = "photorealistic"
    CINEMATIC = "cinematic" 
    ARTISTIC = "artistic"
    FANTASY = "fantasy"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ANIME = "anime"
    OIL_PAINTING = "oil_painting"
    CONCEPT_ART = "concept_art"
    BOOK_COVER = "book_cover"


class ImageProvider(Enum):
    """Image generation providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    STABILITY = "stability"


@dataclass
class OptimizedPrompt:
    """Optimized prompt for image generation"""
    main_prompt: str
    negative_prompt: Optional[str]
    style_modifiers: List[str]
    quality_modifiers: List[str]
    provider_specific: Dict[str, str]
    confidence_score: float
    suggested_parameters: Dict[str, Any]


@dataclass
class PromptTemplate:
    """Template for generating prompts"""
    prefix: str
    core_template: str
    suffix: str
    style_keywords: List[str]
    quality_keywords: List[str]
    negative_keywords: List[str]


class ImagePromptClassifier:
    """Classifies and optimizes image generation prompts"""
    
    def __init__(self, models_dir: str = "models"):
        self.model_manager = create_model_manager()
        self.models_dir = Path(models_dir)
        
        # Cache directory for prompt templates
        self.cache_dir = Path("data/processed/prompts")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Load style templates
        self.style_templates = self._load_style_templates()
        self.provider_templates = self._load_provider_templates()
        
        # Character trait to visual mapping
        self.trait_visuals = self._load_trait_visual_mapping()
        
        # Model placeholders (loaded on demand)
        self._text_analyzer = None
        self._embedding_model = None
        
        logger.info("Image prompt classifier initialized")
    
    def _load_style_templates(self) -> Dict[ImageStyle, PromptTemplate]:
        """Load prompt templates for different image styles"""
        templates = {
            ImageStyle.PHOTOREALISTIC: PromptTemplate(
                prefix="professional photograph,",
                core_template="{subject}, {description}",
                suffix="photorealistic, high detail, sharp focus, 8k resolution",
                style_keywords=["photograph", "realistic", "detailed", "sharp"],
                quality_keywords=["masterpiece", "best quality", "highly detailed", "professional"],
                negative_keywords=["cartoon", "anime", "painting", "sketch", "low quality", "blurry"]
            ),
            
            ImageStyle.CINEMATIC: PromptTemplate(
                prefix="cinematic shot,",
                core_template="{subject}, {description}",
                suffix="dramatic lighting, movie still, high production value, film grain",
                style_keywords=["cinematic", "dramatic", "movie", "film"],
                quality_keywords=["epic", "dramatic lighting", "wide angle", "composition"],
                negative_keywords=["amateur", "low budget", "phone camera", "poor lighting"]
            ),
            
            ImageStyle.ARTISTIC: PromptTemplate(
                prefix="digital art,",
                core_template="{subject}, {description}",
                suffix="concept art, detailed illustration, artistic rendering",
                style_keywords=["digital art", "illustration", "artistic", "creative"],
                quality_keywords=["masterpiece", "detailed", "beautiful", "stunning"],
                negative_keywords=["photograph", "realistic", "3d render", "low quality"]
            ),
            
            ImageStyle.FANTASY: PromptTemplate(
                prefix="fantasy art,",
                core_template="{subject}, {description}",
                suffix="magical atmosphere, detailed fantasy illustration, mystical",
                style_keywords=["fantasy", "magical", "mystical", "enchanted"],
                quality_keywords=["epic fantasy", "detailed", "atmospheric", "otherworldly"],
                negative_keywords=["modern", "realistic", "contemporary", "mundane"]
            ),
            
            ImageStyle.PORTRAIT: PromptTemplate(
                prefix="professional portrait,",
                core_template="portrait of {subject}, {description}",
                suffix="detailed facial features, studio lighting, high quality",
                style_keywords=["portrait", "face", "expression", "character"],
                quality_keywords=["detailed face", "expressive eyes", "perfect anatomy", "professional"],
                negative_keywords=["full body", "landscape", "background focus", "multiple people"]
            ),
            
            ImageStyle.ANIME: PromptTemplate(
                prefix="anime style,",
                core_template="{subject}, {description}",
                suffix="manga art, detailed anime illustration, cel shading",
                style_keywords=["anime", "manga", "japanese art", "cel shaded"],
                quality_keywords=["detailed anime", "beautiful", "expressive", "vibrant colors"],
                negative_keywords=["realistic", "photograph", "3d", "western art"]
            ),
            
            ImageStyle.BOOK_COVER: PromptTemplate(
                prefix="professional book cover design,",
                core_template="{subject}, {description}",
                suffix="artistic rendering, typography space, marketing quality, eye-catching",
                style_keywords=["book cover", "commercial", "marketable", "title space"],
                quality_keywords=["professional design", "high quality", "attractive", "bestseller style"],
                negative_keywords=["amateur", "cluttered", "low quality", "unprofessional"]
            )
        }
        
        return templates
    
    def _load_provider_templates(self) -> Dict[ImageProvider, Dict[str, Any]]:
        """Load provider-specific optimizations"""
        return {
            ImageProvider.OPENAI: {
                "max_prompt_length": 1000,
                "style_preference": "detailed_descriptions",
                "quality_boosters": ["high quality", "detailed", "professional"],
                "avoid_keywords": ["nsfw", "violence", "explicit"],
                "size_recommendations": ["1024x1024", "1792x1024", "1024x1792"]
            },
            
            ImageProvider.HUGGINGFACE: {
                "max_prompt_length": 500,
                "style_preference": "concise_keywords",
                "quality_boosters": ["masterpiece", "best quality", "highly detailed"],
                "negative_prompt_support": True,
                "recommended_steps": 20,
                "guidance_scale": 7.5
            },
            
            ImageProvider.STABILITY: {
                "max_prompt_length": 2000,
                "style_preference": "artistic_keywords",
                "quality_boosters": ["award winning", "trending on artstation", "8k"],
                "negative_prompt_support": True,
                "recommended_steps": 30,
                "guidance_scale": 8.0
            },
            
            ImageProvider.GEMINI: {
                "max_prompt_length": 1500,
                "style_preference": "natural_language",
                "quality_boosters": ["high quality", "detailed", "beautiful"],
                "prompt_enhancement": True  # Can enhance prompts using Gemini
            }
        }
    
    def _load_trait_visual_mapping(self) -> Dict[str, List[str]]:
        """Load mapping from character traits to visual descriptors"""
        return {
            # Personality traits
            "extraversion": ["confident pose", "bright expression", "engaging eyes", "open posture"],
            "agreeableness": ["warm smile", "kind eyes", "gentle expression", "approachable"],
            "conscientiousness": ["neat appearance", "formal attire", "organized", "precise"],
            "neuroticism": ["worried expression", "tense posture", "anxious eyes", "stressed"],
            "openness": ["creative style", "artistic elements", "unique fashion", "expressive"],
            "heroism": ["noble bearing", "strong jawline", "determined expression", "heroic pose"],
            "intelligence": ["thoughtful expression", "glasses", "scholarly appearance", "wise eyes"],
            "courage": ["confident stance", "fearless expression", "bold posture", "brave"],
            
            # Physical attributes
            "tall": ["towering", "imposing height", "long limbs", "statuesque"],
            "short": ["petite", "compact build", "small stature", "diminutive"],
            "muscular": ["athletic build", "strong physique", "well-built", "powerful frame"],
            "slim": ["slender figure", "lean build", "graceful", "elegant proportions"],
            
            # Age groups
            "child": ["youthful face", "innocent eyes", "small features", "childlike"],
            "teenager": ["young face", "adolescent features", "youthful energy", "teenage"],
            "adult": ["mature features", "adult proportions", "developed", "grown"],
            "elderly": ["aged features", "wise expression", "weathered", "senior"]
        }
    
    def optimize_character_portrait_prompt(
        self,
        character: Character,
        style: ImageStyle = ImageStyle.PORTRAIT,
        provider: ImageProvider = ImageProvider.HUGGINGFACE
    ) -> OptimizedPrompt:
        """Optimize prompt for character portrait generation"""
        
        # Extract visual descriptors from character
        visual_elements = self._extract_character_visuals(character)
        
        # Build core prompt
        subject = f"{character.name}"
        description = self._build_character_description(character, visual_elements)
        
        # Get style template
        template = self.style_templates.get(style, self.style_templates[ImageStyle.PORTRAIT])
        
        # Build main prompt
        core_prompt = template.core_template.format(
            subject=subject,
            description=description
        )
        
        main_prompt = f"{template.prefix} {core_prompt}, {', '.join(template.style_keywords)}, {template.suffix}"
        
        # Build negative prompt
        negative_prompt = ", ".join(template.negative_keywords)
        
        # Add provider-specific optimizations
        provider_config = self.provider_templates.get(provider, {})
        main_prompt = self._apply_provider_optimizations(main_prompt, provider_config)
        
        # Calculate confidence score
        confidence = self._calculate_prompt_confidence(character, visual_elements)
        
        # Suggested parameters
        suggested_params = self._get_suggested_parameters(style, provider)
        
        return OptimizedPrompt(
            main_prompt=main_prompt,
            negative_prompt=negative_prompt,
            style_modifiers=template.style_keywords,
            quality_modifiers=template.quality_keywords,
            provider_specific={provider.value: main_prompt},
            confidence_score=confidence,
            suggested_parameters=suggested_params
        )
    
    def optimize_scene_image_prompt(
        self,
        scene: Scene,
        style: ImageStyle = ImageStyle.CINEMATIC,
        provider: ImageProvider = ImageProvider.HUGGINGFACE
    ) -> OptimizedPrompt:
        """Optimize prompt for scene image generation"""
        
        # Extract scene elements
        scene_elements = self._extract_scene_visuals(scene)
        
        # Build scene description
        subject = scene_elements.get("location", "scene")
        description = self._build_scene_description(scene, scene_elements)
        
        # Get style template
        template = self.style_templates.get(style, self.style_templates[ImageStyle.CINEMATIC])
        
        # Build main prompt
        core_prompt = template.core_template.format(
            subject=subject,
            description=description
        )
        
        main_prompt = f"{template.prefix} {core_prompt}, {', '.join(template.style_keywords)}, {template.suffix}"
        
        # Build negative prompt
        negative_prompt = ", ".join(template.negative_keywords)
        
        # Add mood-specific elements
        if hasattr(scene, 'mood') and scene.mood:
            mood_modifiers = self._get_mood_modifiers(scene.mood)
            main_prompt += f", {', '.join(mood_modifiers)}"
        
        # Apply provider optimizations
        provider_config = self.provider_templates.get(provider, {})
        main_prompt = self._apply_provider_optimizations(main_prompt, provider_config)
        
        # Calculate confidence score
        confidence = self._calculate_scene_prompt_confidence(scene, scene_elements)
        
        # Suggested parameters
        suggested_params = self._get_suggested_parameters(style, provider)
        
        return OptimizedPrompt(
            main_prompt=main_prompt,
            negative_prompt=negative_prompt,
            style_modifiers=template.style_keywords,
            quality_modifiers=template.quality_keywords,
            provider_specific={provider.value: main_prompt},
            confidence_score=confidence,
            suggested_parameters=suggested_params
        )
    
    def _extract_character_visuals(self, character: Character) -> Dict[str, List[str]]:
        """Extract visual elements from character data"""
        visuals = {
            "personality_visuals": [],
            "physical_visuals": [],
            "age_visuals": [],
            "style_visuals": []
        }
        
        # Extract from personality
        if hasattr(character, 'personality') and character.personality:
            personality = character.personality
            for trait, value in personality.dict().items():
                if trait != 'confidence' and value > 0.6:  # Strong traits only
                    if trait in self.trait_visuals:
                        visuals["personality_visuals"].extend(self.trait_visuals[trait])
        
        # Extract from physical description
        if hasattr(character, 'physical_description') and character.physical_description:
            phys = character.physical_description
            
            # Height and build
            if hasattr(phys, 'height') and phys.height:
                if phys.height in self.trait_visuals:
                    visuals["physical_visuals"].extend(self.trait_visuals[phys.height])
            
            if hasattr(phys, 'build') and phys.build:
                if phys.build in self.trait_visuals:
                    visuals["physical_visuals"].extend(self.trait_visuals[phys.build])
            
            # Hair and eyes
            if hasattr(phys, 'hair_color') and phys.hair_color:
                visuals["physical_visuals"].append(f"{phys.hair_color} hair")
            
            if hasattr(phys, 'eye_color') and phys.eye_color:
                visuals["physical_visuals"].append(f"{phys.eye_color} eyes")
            
            # Distinctive features
            if hasattr(phys, 'distinctive_features') and phys.distinctive_features:
                visuals["physical_visuals"].extend(phys.distinctive_features)
            
            # Clothing style
            if hasattr(phys, 'clothing_style') and phys.clothing_style:
                visuals["style_visuals"].append(f"{phys.clothing_style} attire")
        
        # Age group visuals
        if hasattr(character, 'age_group') and character.age_group:
            age_group = character.age_group.value if hasattr(character.age_group, 'value') else str(character.age_group)
            if age_group in self.trait_visuals:
                visuals["age_visuals"].extend(self.trait_visuals[age_group])
        
        return visuals
    
    def _build_character_description(self, character: Character, visual_elements: Dict[str, List[str]]) -> str:
        """Build character description from visual elements"""
        description_parts = []
        
        # Add physical visuals
        if visual_elements["physical_visuals"]:
            description_parts.extend(visual_elements["physical_visuals"][:3])  # Limit to top 3
        
        # Add personality visuals (subtle)
        if visual_elements["personality_visuals"]:
            description_parts.extend(visual_elements["personality_visuals"][:2])  # Limit to top 2
        
        # Add age visuals
        if visual_elements["age_visuals"]:
            description_parts.extend(visual_elements["age_visuals"][:1])  # Just one
        
        # Add style visuals
        if visual_elements["style_visuals"]:
            description_parts.extend(visual_elements["style_visuals"][:1])
        
        return ", ".join(description_parts)
    
    def _extract_scene_visuals(self, scene: Scene) -> Dict[str, Any]:
        """Extract visual elements from scene data"""
        visuals = {
            "location": "scene",
            "atmosphere": [],
            "lighting": [],
            "mood": [],
            "elements": []
        }
        
        # Extract from setting
        if hasattr(scene, 'setting') and scene.setting:
            setting = scene.setting
            
            # Location name
            if hasattr(setting, 'name') and setting.name:
                visuals["location"] = setting.name
            
            # Setting type
            if hasattr(setting, 'type') and setting.type:
                visuals["elements"].append(setting.type)
            
            # Description
            if hasattr(setting, 'description') and setting.description:
                visuals["elements"].append(setting.description)
            
            # Atmosphere
            if hasattr(setting, 'atmosphere') and setting.atmosphere:
                visuals["atmosphere"].append(setting.atmosphere)
        
        # Extract from narrative summary
        if hasattr(scene, 'narrative_summary') and scene.narrative_summary:
            # Simple keyword extraction from narrative
            narrative = scene.narrative_summary.lower()
            
            # Look for lighting keywords
            lighting_words = ["bright", "dark", "dim", "sunny", "moonlight", "candlelight", "firelight"]
            for word in lighting_words:
                if word in narrative:
                    visuals["lighting"].append(word)
            
            # Look for atmosphere keywords
            atmosphere_words = ["peaceful", "tense", "mysterious", "cheerful", "gloomy", "magical"]
            for word in atmosphere_words:
                if word in narrative:
                    visuals["atmosphere"].append(word)
        
        return visuals
    
    def _build_scene_description(self, scene: Scene, scene_elements: Dict[str, Any]) -> str:
        """Build scene description from visual elements"""
        description_parts = []
        
        # Add atmosphere
        if scene_elements["atmosphere"]:
            description_parts.extend(scene_elements["atmosphere"][:2])
        
        # Add lighting
        if scene_elements["lighting"]:
            description_parts.extend(scene_elements["lighting"][:2])
        
        # Add other elements
        if scene_elements["elements"]:
            description_parts.extend(scene_elements["elements"][:3])
        
        return ", ".join(description_parts)
    
    def _get_mood_modifiers(self, mood: str) -> List[str]:
        """Get visual modifiers for a given mood"""
        mood_mapping = {
            "dark": ["dark atmosphere", "moody lighting", "shadows", "dramatic"],
            "bright": ["bright lighting", "cheerful", "vibrant colors", "uplifting"],
            "mysterious": ["mysterious atmosphere", "fog", "dramatic shadows", "enigmatic"],
            "peaceful": ["serene", "calm", "soft lighting", "tranquil"],
            "tense": ["dramatic tension", "sharp contrasts", "intense", "suspenseful"],
            "romantic": ["soft lighting", "warm colors", "intimate", "dreamy"],
            "action": ["dynamic", "motion blur", "intense", "energetic"]
        }
        
        return mood_mapping.get(mood.lower(), ["atmospheric"])
    
    def _apply_provider_optimizations(self, prompt: str, provider_config: Dict[str, Any]) -> str:
        """Apply provider-specific optimizations"""
        optimized_prompt = prompt
        
        # Length optimization
        max_length = provider_config.get("max_prompt_length", 1000)
        if len(optimized_prompt) > max_length:
            # Truncate while preserving structure
            words = optimized_prompt.split(", ")
            truncated_words = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 2 <= max_length:  # +2 for ", "
                    truncated_words.append(word)
                    current_length += len(word) + 2
                else:
                    break
            
            optimized_prompt = ", ".join(truncated_words)
        
        # Add quality boosters
        quality_boosters = provider_config.get("quality_boosters", [])
        if quality_boosters:
            optimized_prompt += ", " + ", ".join(quality_boosters[:2])  # Add top 2 boosters
        
        return optimized_prompt
    
    def _calculate_prompt_confidence(self, character: Character, visual_elements: Dict[str, List[str]]) -> float:
        """Calculate confidence score for character prompt"""
        confidence_factors = []
        
        # Factor 1: Amount of visual information available
        total_visuals = sum(len(visuals) for visuals in visual_elements.values())
        visual_confidence = min(total_visuals / 10.0, 1.0)  # Up to 10 visual elements
        confidence_factors.append(visual_confidence)
        
        # Factor 2: Completeness of character data
        data_completeness = 0.0
        if hasattr(character, 'personality') and character.personality:
            data_completeness += 0.4
        if hasattr(character, 'physical_description') and character.physical_description:
            data_completeness += 0.4
        if hasattr(character, 'name') and character.name:
            data_completeness += 0.2
        confidence_factors.append(data_completeness)
        
        # Factor 3: Quality of personality data (if available)
        if hasattr(character, 'personality') and character.personality:
            personality_confidence = getattr(character.personality, 'confidence', 0.5)
            confidence_factors.append(personality_confidence)
        
        # Return average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_scene_prompt_confidence(self, scene: Scene, scene_elements: Dict[str, Any]) -> float:
        """Calculate confidence score for scene prompt"""
        confidence_factors = []
        
        # Factor 1: Amount of scene information
        total_elements = sum(len(elements) if isinstance(elements, list) else 1 
                           for elements in scene_elements.values() if elements)
        element_confidence = min(total_elements / 8.0, 1.0)
        confidence_factors.append(element_confidence)
        
        # Factor 2: Completeness of scene data
        data_completeness = 0.0
        if hasattr(scene, 'setting') and scene.setting:
            data_completeness += 0.4
        if hasattr(scene, 'narrative_summary') and scene.narrative_summary:
            data_completeness += 0.4
        if hasattr(scene, 'mood') and scene.mood:
            data_completeness += 0.2
        confidence_factors.append(data_completeness)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _get_suggested_parameters(self, style: ImageStyle, provider: ImageProvider) -> Dict[str, Any]:
        """Get suggested generation parameters"""
        base_params = {}
        
        # Style-specific parameters
        if style == ImageStyle.PHOTOREALISTIC:
            base_params.update({
                "guidance_scale": 7.0,
                "steps": 25,
                "quality": "hd"
            })
        elif style == ImageStyle.ARTISTIC:
            base_params.update({
                "guidance_scale": 8.5,
                "steps": 30,
                "quality": "standard"
            })
        elif style == ImageStyle.ANIME:
            base_params.update({
                "guidance_scale": 9.0,
                "steps": 28,
                "quality": "standard"
            })
        
        # Provider-specific parameters
        provider_config = self.provider_templates.get(provider, {})
        if "recommended_steps" in provider_config:
            base_params["steps"] = provider_config["recommended_steps"]
        if "guidance_scale" in provider_config:
            base_params["guidance_scale"] = provider_config["guidance_scale"]
        
        return base_params
    
    def enhance_prompt_with_ai(self, base_prompt: str, style: ImageStyle) -> str:
        """Use AI to enhance the prompt (if available)"""
        if not TRANSFORMERS_AVAILABLE:
            return base_prompt
        
        # This is a placeholder for AI-based prompt enhancement
        # You could use models like GPT-3.5 or Gemini to improve prompts
        
        try:
            # Simple enhancement by adding style-appropriate terms
            style_enhancers = {
                ImageStyle.PHOTOREALISTIC: ["ultra-detailed", "sharp focus", "professional photography"],
                ImageStyle.ARTISTIC: ["masterpiece", "trending on artstation", "beautiful composition"],
                ImageStyle.FANTASY: ["epic fantasy", "magical realism", "ethereal beauty"],
                ImageStyle.CINEMATIC: ["epic cinematography", "dramatic composition", "film grain"]
            }
            
            enhancers = style_enhancers.get(style, ["high quality", "detailed"])
            enhanced_prompt = f"{base_prompt}, {', '.join(enhancers)}"
            
            return enhanced_prompt
            
        except Exception as e:
            logger.warning(f"AI prompt enhancement failed: {e}")
            return base_prompt
    
    def save_prompt_template(self, name: str, template: PromptTemplate):
        """Save a custom prompt template"""
        try:
            template_file = self.cache_dir / f"{name}_template.json"
            
            template_data = {
                "name": name,
                "prefix": template.prefix,
                "core_template": template.core_template,
                "suffix": template.suffix,
                "style_keywords": template.style_keywords,
                "quality_keywords": template.quality_keywords,
                "negative_keywords": template.negative_keywords
            }
            
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Prompt template saved: {template_file}")
            
        except Exception as e:
            logger.error(f"Failed to save prompt template: {e}")
    
    def load_custom_template(self, name: str) -> Optional[PromptTemplate]:
        """Load a custom prompt template"""
        try:
            template_file = self.cache_dir / f"{name}_template.json"
            
            if template_file.exists():
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                return PromptTemplate(
                    prefix=template_data["prefix"],
                    core_template=template_data["core_template"],
                    suffix=template_data["suffix"],
                    style_keywords=template_data["style_keywords"],
                    quality_keywords=template_data["quality_keywords"],
                    negative_keywords=template_data["negative_keywords"]
                )
            
        except Exception as e:
            logger.error(f"Failed to load custom template: {e}")
        
        return None


def create_prompt_classifier() -> ImagePromptClassifier:
    """Create and configure the image prompt classifier"""
    return ImagePromptClassifier()


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..core.models import Character, PersonalityVector, PhysicalDescription, Gender, AgeGroup
    
    async def test_prompt_classifier():
        """Test the prompt classifier functionality"""
        classifier = create_prompt_classifier()
        
        # Create test character
        test_character = Character(
            character_id="test_char_1",
            name="Elena Shadowmere",
            booknlp_id=1,
            gender=Gender.FEMALE,
            age_group=AgeGroup.ADULT,
            personality=PersonalityVector(
                extraversion=0.3,
                agreeableness=0.7,
                conscientiousness=0.8,
                neuroticism=0.2,
                openness=0.9,
                heroism=0.8,
                intelligence=0.9,
                confidence=0.85
            ),
            physical_description=PhysicalDescription(
                height="tall",
                build="slender",
                hair_color="dark",
                eye_color="green",
                age_appearance="adult",
                distinctive_features=["scar on left cheek"],
                clothing_style="fantasy"
            ),
            importance_score=0.95,
            first_appearance=1,
            last_appearance=10,
            total_dialogue_lines=50
        )
        
        print("\n=== Testing Character Portrait Generation ===")
        
        # Test different styles and providers
        test_configs = [
            (ImageStyle.PORTRAIT, ImageProvider.HUGGINGFACE),
            (ImageStyle.FANTASY, ImageProvider.STABILITY),
            (ImageStyle.CINEMATIC, ImageProvider.OPENAI),
            (ImageStyle.ARTISTIC, ImageProvider.GEMINI)
        ]
        
        for style, provider in test_configs:
            optimized = classifier.optimize_character_portrait_prompt(
                test_character, style, provider
            )
            
            print(f"\n--- {style.value.upper()} style with {provider.value} ---")
            print(f"Main Prompt: {optimized.main_prompt}")
            print(f"Negative Prompt: {optimized.negative_prompt}")
            print(f"Confidence: {optimized.confidence_score:.2f}")
            print(f"Suggested Parameters: {optimized.suggested_parameters}")
    
    asyncio.run(test_prompt_classifier())