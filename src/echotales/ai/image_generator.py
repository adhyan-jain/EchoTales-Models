"""
Image Generation Service for EchoTales

This module provides image generation capabilities using multiple AI APIs:
- Google Gemini (free tier with good limits)
- OpenAI DALL-E (limited use)
- HuggingFace Inference API (free tier)
- Stability AI (if API key available)

The service implements fallback mechanisms to ensure reliable image generation.
"""

import os
import requests
import base64
import time
import logging
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageProvider(Enum):
    """Available image generation providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    STABILITY = "stability"

@dataclass
class ImageRequest:
    """Image generation request parameters"""
    prompt: str
    style: str = "photorealistic"
    size: str = "1024x1024"
    quality: str = "standard"
    model: str = "auto"
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: int = 20
    guidance_scale: float = 7.5


@dataclass
class ImageResult:
    """Image generation result"""
    success: bool
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    provider: Optional[str] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None


class ImageGenerationService:
    """Multi-provider image generation service"""
    
    def __init__(self, output_dir: str = "data/output/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys from environment
        self.api_keys = {
            'gemini': os.getenv('GEMINI_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_API_KEY'),
            'stability': os.getenv('STABILITY_API_KEY')
        }
        
        # Provider priority (free/low-cost first)
        self.provider_priority = [
            ImageProvider.GEMINI,      # Free with good limits
            ImageProvider.HUGGINGFACE, # Free tier available
            ImageProvider.OPENAI,      # Paid but high quality
            ImageProvider.STABILITY    # Paid but excellent quality
        ]
        
        # Provider configurations
        self.provider_configs = {
            ImageProvider.GEMINI: {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'model': 'imagen-3.0-generate-001',  # Gemini's image model
                'max_requests_per_minute': 15,
                'free_tier': True
            },
            ImageProvider.OPENAI: {
                'base_url': 'https://api.openai.com/v1',
                'model': 'dall-e-3',
                'max_requests_per_minute': 5,
                'free_tier': False
            },
            ImageProvider.HUGGINGFACE: {
                'base_url': 'https://api-inference.huggingface.co/models',
                'model': 'stabilityai/stable-diffusion-xl-base-1.0',
                'max_requests_per_minute': 10,
                'free_tier': True
            },
            ImageProvider.STABILITY: {
                'base_url': 'https://api.stability.ai/v1',
                'model': 'stable-diffusion-v1-6',
                'max_requests_per_minute': 20,
                'free_tier': False
            }
        }
        
        # Request tracking for rate limiting
        self.request_history = {}
        
        logger.info("Image generation service initialized")
        self._log_available_providers()
    
    def _log_available_providers(self):
        """Log which providers are available"""
        available = []
        for provider in ImageProvider:
            if self.api_keys.get(provider.value):
                available.append(provider.value)
            else:
                logger.warning(f"API key not found for {provider.value}")
        
        logger.info(f"Available providers: {', '.join(available) if available else 'None'}")
        
        if not available:
            logger.warning("No API keys found. Image generation will not work.")
    
    async def generate_image(self, request: ImageRequest) -> ImageResult:
        """Generate image using available providers with fallback"""
        
        if not any(self.api_keys.values()):
            return ImageResult(
                success=False,
                error="No API keys configured for image generation"
            )
        
        # Enhance prompt based on style
        enhanced_prompt = self._enhance_prompt(request.prompt, request.style)
        request.prompt = enhanced_prompt
        
        # Try providers in priority order
        for provider in self.provider_priority:
            if not self.api_keys.get(provider.value):
                continue
            
            if not self._can_make_request(provider):
                logger.warning(f"Rate limit exceeded for {provider.value}, trying next provider")
                continue
            
            try:
                logger.info(f"Attempting image generation with {provider.value}")
                result = await self._generate_with_provider(provider, request)
                
                if result.success:
                    logger.info(f"Successfully generated image with {provider.value}")
                    self._record_request(provider)
                    return result
                else:
                    logger.warning(f"Failed with {provider.value}: {result.error}")
            
            except Exception as e:
                logger.error(f"Error with provider {provider.value}: {e}")
                continue
        
        return ImageResult(
            success=False,
            error="All image generation providers failed"
        )
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Enhance prompt based on style preferences"""
        style_prefixes = {
            "photorealistic": "professional photograph, photorealistic, high detail, sharp focus,",
            "cinematic": "cinematic shot, dramatic lighting, movie still, high production value,",
            "artistic": "digital art, concept art, detailed illustration, artistic rendering,",
            "fantasy": "fantasy art, magical atmosphere, detailed fantasy illustration,",
            "portrait": "professional portrait, detailed facial features, studio lighting,",
            "landscape": "landscape photography, scenic view, natural lighting, wide angle,",
            "anime": "anime style, manga art, detailed anime illustration,",
            "oil_painting": "oil painting style, classical art, painted texture, artistic brushstrokes,"
        }
        
        prefix = style_prefixes.get(style.lower(), "high quality digital art,")
        suffix = "masterpiece, best quality, highly detailed"
        
        return f"{prefix} {prompt}, {suffix}"
    
    def _can_make_request(self, provider: ImageProvider) -> bool:
        """Check if we can make a request to the provider (rate limiting)"""
        current_time = time.time()
        provider_key = provider.value
        
        if provider_key not in self.request_history:
            self.request_history[provider_key] = []
        
        # Remove old requests (older than 1 minute)
        cutoff_time = current_time - 60
        self.request_history[provider_key] = [
            req_time for req_time in self.request_history[provider_key]
            if req_time > cutoff_time
        ]
        
        # Check if under limit
        max_requests = self.provider_configs[provider]['max_requests_per_minute']
        current_requests = len(self.request_history[provider_key])
        
        return current_requests < max_requests
    
    def _record_request(self, provider: ImageProvider):
        """Record a request for rate limiting"""
        current_time = time.time()
        provider_key = provider.value
        
        if provider_key not in self.request_history:
            self.request_history[provider_key] = []
        
        self.request_history[provider_key].append(current_time)
    
    async def _generate_with_provider(self, provider: ImageProvider, request: ImageRequest) -> ImageResult:
        """Generate image with specific provider"""
        
        if provider == ImageProvider.GEMINI:
            return await self._generate_with_gemini(request)
        elif provider == ImageProvider.OPENAI:
            return await self._generate_with_openai(request)
        elif provider == ImageProvider.HUGGINGFACE:
            return await self._generate_with_huggingface(request)
        elif provider == ImageProvider.STABILITY:
            return await self._generate_with_stability(request)
        else:
            return ImageResult(success=False, error=f"Unknown provider: {provider}")
    
    async def _generate_with_gemini(self, request: ImageRequest) -> ImageResult:
        """Generate image using Google Gemini"""
        try:
            # Note: As of 2024, Gemini doesn't have direct image generation
            # But it can be used to enhance prompts for other services
            # For now, we'll use it to enhance the prompt and pass to another service
            
            # First, enhance the prompt with Gemini
            enhanced_prompt = await self._enhance_prompt_with_gemini(request.prompt)
            
            # Then use HuggingFace with the enhanced prompt
            if self.api_keys.get('huggingface'):
                enhanced_request = ImageRequest(
                    prompt=enhanced_prompt,
                    style=request.style,
                    size=request.size,
                    quality=request.quality
                )
                result = await self._generate_with_huggingface(enhanced_request)
                if result.success:
                    result.provider = "gemini_enhanced"
                return result
            
            return ImageResult(
                success=False,
                error="Gemini image generation not available, and no fallback provider"
            )
            
        except Exception as e:
            return ImageResult(success=False, error=f"Gemini generation failed: {e}")
    
    async def _enhance_prompt_with_gemini(self, prompt: str) -> str:
        """Use Gemini to enhance the image prompt"""
        try:
            api_key = self.api_keys['gemini']
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            
            enhance_prompt = f"""
            Please enhance this image generation prompt to be more detailed and specific for AI image generation.
            Add artistic details, lighting, composition, and style information.
            Keep the core concept but make it more descriptive.
            
            Original prompt: {prompt}
            
            Enhanced prompt:"""
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": enhance_prompt
                    }]
                }]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    enhanced = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    logger.info(f"Gemini enhanced prompt: {enhanced}")
                    return enhanced
            
            # Fallback to original prompt
            return prompt
            
        except Exception as e:
            logger.warning(f"Failed to enhance prompt with Gemini: {e}")
            return prompt
    
    async def _generate_with_openai(self, request: ImageRequest) -> ImageResult:
        """Generate image using OpenAI DALL-E"""
        try:
            api_key = self.api_keys['openai']
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "dall-e-3",
                "prompt": request.prompt,
                "n": 1,
                "size": request.size,
                "quality": request.quality,
                "response_format": "url"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result and result['data']:
                    image_url = result['data'][0]['url']
                    
                    # Download and save the image
                    image_path = await self._download_and_save_image(image_url, "openai")
                    
                    return ImageResult(
                        success=True,
                        image_path=image_path,
                        image_url=image_url,
                        provider="openai",
                        metadata={
                            "model": "dall-e-3",
                            "revised_prompt": result['data'][0].get('revised_prompt', request.prompt)
                        }
                    )
            
            return ImageResult(
                success=False,
                error=f"OpenAI API error: {response.status_code} - {response.text}"
            )
            
        except Exception as e:
            return ImageResult(success=False, error=f"OpenAI generation failed: {e}")
    
    async def _generate_with_huggingface(self, request: ImageRequest) -> ImageResult:
        """Generate image using HuggingFace Inference API"""
        try:
            api_key = self.api_keys['huggingface']
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Use Stable Diffusion XL model
            model_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            
            payload = {
                "inputs": request.prompt,
                "parameters": {
                    "guidance_scale": request.guidance_scale,
                    "num_inference_steps": request.steps,
                    "seed": request.seed
                }
            }
            
            if request.negative_prompt:
                payload["parameters"]["negative_prompt"] = request.negative_prompt
            
            response = requests.post(model_url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                # HuggingFace returns binary image data
                image_data = response.content
                
                # Save the image
                image_path = await self._save_image_data(image_data, "huggingface")
                
                return ImageResult(
                    success=True,
                    image_path=image_path,
                    provider="huggingface",
                    metadata={
                        "model": "stabilityai/stable-diffusion-xl-base-1.0",
                        "guidance_scale": request.guidance_scale,
                        "steps": request.steps
                    }
                )
            
            return ImageResult(
                success=False,
                error=f"HuggingFace API error: {response.status_code} - {response.text}"
            )
            
        except Exception as e:
            return ImageResult(success=False, error=f"HuggingFace generation failed: {e}")
    
    async def _generate_with_stability(self, request: ImageRequest) -> ImageResult:
        """Generate image using Stability AI"""
        try:
            api_key = self.api_keys['stability']
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Parse size
            width, height = map(int, request.size.split('x'))
            
            payload = {
                "text_prompts": [
                    {"text": request.prompt, "weight": 1.0}
                ],
                "cfg_scale": request.guidance_scale,
                "steps": request.steps,
                "width": width,
                "height": height,
                "samples": 1,
                "sampler": "K_DPM_2_ANCESTRAL"
            }
            
            if request.negative_prompt:
                payload["text_prompts"].append({
                    "text": request.negative_prompt,
                    "weight": -1.0
                })
            
            if request.seed:
                payload["seed"] = request.seed
            
            response = requests.post(
                "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image",
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'artifacts' in result and result['artifacts']:
                    # Decode base64 image
                    image_b64 = result['artifacts'][0]['base64']
                    image_data = base64.b64decode(image_b64)
                    
                    # Save the image
                    image_path = await self._save_image_data(image_data, "stability")
                    
                    return ImageResult(
                        success=True,
                        image_path=image_path,
                        provider="stability",
                        metadata={
                            "model": "stable-diffusion-v1-6",
                            "cfg_scale": request.guidance_scale,
                            "steps": request.steps,
                            "seed": result['artifacts'][0].get('seed')
                        }
                    )
            
            return ImageResult(
                success=False,
                error=f"Stability API error: {response.status_code} - {response.text}"
            )
            
        except Exception as e:
            return ImageResult(success=False, error=f"Stability generation failed: {e}")
    
    async def _download_and_save_image(self, image_url: str, provider: str) -> str:
        """Download image from URL and save locally"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            return await self._save_image_data(response.content, provider)
            
        except Exception as e:
            raise Exception(f"Failed to download image: {e}")
    
    async def _save_image_data(self, image_data: bytes, provider: str) -> str:
        """Save image data to local file"""
        try:
            # Generate unique filename
            image_hash = hashlib.md5(image_data).hexdigest()[:12]
            filename = f"{provider}_{int(time.time())}_{image_hash}.jpg"
            image_path = self.output_dir / filename
            
            # Convert and save as JPEG
            image = Image.open(io.BytesIO(image_data))
            
            # Convert RGBA to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # Save with high quality
            image.save(image_path, 'JPEG', quality=95, optimize=True)
            
            logger.info(f"Image saved to {image_path}")
            return str(image_path)
            
        except Exception as e:
            raise Exception(f"Failed to save image: {e}")
    
    def generate_character_portrait(self, character_name: str, description: str, 
                                  style: str = "portrait") -> ImageRequest:
        """Generate optimized prompt for character portrait"""
        prompt = f"Portrait of {character_name}, {description}"
        
        return ImageRequest(
            prompt=prompt,
            style=style,
            size="1024x1024",
            quality="hd" if "openai" in str(self.provider_priority) else "standard"
        )
    
    def generate_scene_image(self, scene_description: str, mood: str = "neutral", 
                            style: str = "cinematic") -> ImageRequest:
        """Generate optimized prompt for scene image"""
        mood_modifiers = {
            "dark": "dark atmosphere, moody lighting, shadows",
            "bright": "bright lighting, cheerful atmosphere, vibrant colors",
            "mysterious": "mysterious atmosphere, fog, dramatic shadows",
            "peaceful": "serene atmosphere, soft lighting, calm mood",
            "intense": "dramatic lighting, intense atmosphere, dynamic composition"
        }
        
        mood_text = mood_modifiers.get(mood.lower(), "")
        prompt = f"{scene_description}, {mood_text}".strip(", ")
        
        return ImageRequest(
            prompt=prompt,
            style=style,
            size="1024x1024"
        )


# Example usage and utility functions
def create_image_service() -> ImageGenerationService:
    """Create and configure image generation service"""
    return ImageGenerationService()


async def generate_book_cover(title: str, author: str, genre: str, 
                            service: ImageGenerationService) -> ImageResult:
    """Generate a book cover image"""
    prompt = f"Book cover for '{title}' by {author}, {genre} genre, professional book cover design, typography space"
    
    request = ImageRequest(
        prompt=prompt,
        style="artistic",
        size="1024x1536"  # Book cover aspect ratio
    )
    
    return await service.generate_image(request)


if __name__ == "__main__":
    import asyncio
    
    async def test_generation():
        service = create_image_service()
        
        # Test character portrait
        request = service.generate_character_portrait(
            "Eleanor", 
            "middle-aged woman with kind eyes, wearing medieval dress",
            "portrait"
        )
        
        result = await service.generate_image(request)
        
        if result.success:
            print(f"✓ Image generated successfully with {result.provider}")
            print(f"  Path: {result.image_path}")
        else:
            print(f"✗ Generation failed: {result.error}")
    
    asyncio.run(test_generation())