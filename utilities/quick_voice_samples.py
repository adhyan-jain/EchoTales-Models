#!/usr/bin/env python3
"""
Quick Character Voice Sample Generator
Generate awesome voice samples for Zhou Mingrui and other characters
"""

import os
import asyncio
from pathlib import Path
import random

# Character voice samples
CHARACTER_SAMPLES = {
    'Zhou Mingrui': [
        "I never expected my life to change so dramatically in a single moment.",
        "The mysteries of this world are far beyond what I initially imagined.", 
        "Sometimes I wonder if I'm truly prepared for what lies ahead.",
        "The path I've chosen is dangerous, but there's no turning back now.",
        "Every decision I make seems to ripple through reality in unexpected ways.",
        "Klein Moretti... that name carries more weight than anyone could imagine.",
        "The fool's path is treacherous, but it's the only way forward.",
        "In this world of beyonders, knowledge is both power and curse."
    ],
    'Narrator': [
        "In the depths of the mysterious city, ancient secrets stirred to life.",
        "The fog rolled in from the harbor, carrying with it whispers of forgotten tales.", 
        "Time seemed to slow as the protagonist faced his greatest challenge yet.",
        "The room fell silent, save for the steady ticking of an antique clock.",
        "Beyond the veil of reality, forces beyond comprehension began to stir.",
        "The old manor creaked under the weight of its dark history.",
        "Shadows danced in the candlelight, revealing glimpses of otherworldly truths."
    ],
    'Klein Moretti': [
        "The crimson moon hangs low tonight, casting eerie shadows across the city.",
        "As a fortune teller, I've learned to read the signs that others miss.",
        "The tarot cards never lie, but their truths are often difficult to accept.",
        "In this age of steam and machinery, the supernatural still holds dominion.",
        "I must be careful - one misstep could expose my true nature."
    ]
}

def ensure_character_dirs():
    """Create character directories"""
    base_dir = Path("generated_audio/characters")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    dirs = {}
    for char in CHARACTER_SAMPLES.keys():
        char_dir = base_dir / char.lower().replace(' ', '_')
        char_dir.mkdir(exist_ok=True)
        dirs[char] = char_dir
    
    return dirs

async def generate_character_samples():
    """Generate character voice samples using Edge TTS"""
    print("🎭 EchoTales Quick Voice Sample Generator")
    print("=" * 45)
    
    try:
        import edge_tts
        
        # Character voice mappings
        voice_mapping = {
            'Zhou Mingrui': 'en-US-ChristopherNeural',  # The awesome voice!
            'Narrator': 'en-US-ChristopherNeural',
            'Klein Moretti': 'en-US-RogerNeural'  # Slightly different for variety
        }
        
        # Create directories
        char_dirs = ensure_character_dirs()
        
        print("🎤 Generating character voice samples...")
        
        for char_name, samples in CHARACTER_SAMPLES.items():
            voice = voice_mapping.get(char_name, 'en-US-AriaNeural')
            char_dir = char_dirs[char_name]
            
            print(f"\n🌟 Creating samples for {char_name} ({voice})")
            
            # Select 4 random samples for variety
            selected_samples = random.sample(samples, min(4, len(samples)))
            
            for i, sample_text in enumerate(selected_samples, 1):
                output_file = char_dir / f"{char_name.lower().replace(' ', '_')}_sample_{i}.mp3"
                
                try:
                    # Create TTS with character-specific personality
                    if char_name == 'Zhou Mingrui':
                        # Cool, confident protagonist voice
                        communicate = edge_tts.Communicate(sample_text, voice, rate="+8%", pitch="+3Hz")
                    elif char_name == 'Narrator':
                        # Deep, authoritative storytelling voice  
                        communicate = edge_tts.Communicate(sample_text, voice, rate="+2%", pitch="-8Hz")
                    else:
                        # Klein - thoughtful, mysterious tone
                        communicate = edge_tts.Communicate(sample_text, voice, rate="+0%", pitch="+2Hz")
                    
                    await communicate.save(str(output_file))
                    print(f"    ✅ Sample {i}: {sample_text[:60]}...")
                    
                except Exception as e:
                    print(f"    ❌ Failed to generate sample {i}: {e}")
        
        # Create a summary JSON
        summary_data = {
            'characters': {
                char: {
                    'voice': voice_mapping.get(char, 'en-US-AriaNeural'),
                    'samples_generated': min(4, len(samples)),
                    'sample_texts': random.sample(samples, min(4, len(samples)))
                }
                for char, samples in CHARACTER_SAMPLES.items()
            },
            'generation_info': {
                'total_characters': len(CHARACTER_SAMPLES),
                'awesome_voice_character': 'Zhou Mingrui',
                'voice_quality': 'High quality with character-specific personality adjustments'
            }
        }
        
        import json
        summary_file = Path("generated_audio/characters/voice_samples_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 SUCCESS! Character voice samples generated!")
        print(f"📁 Generated content in generated_audio/characters/:")
        
        for char_name in CHARACTER_SAMPLES.keys():
            char_folder = char_name.lower().replace(' ', '_')
            print(f"   📂 {char_folder}/ - 4 voice samples")
        
        print(f"   📄 voice_samples_summary.json - Complete summary")
        print(f"\n🌟 Zhou Mingrui samples feature that awesome ChristopherNeural voice!")
        
        return True
        
    except ImportError:
        print("❌ edge-tts not available. Install with: pip install edge-tts")
        return False
    except Exception as e:
        print(f"❌ Failed to generate samples: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(generate_character_samples())