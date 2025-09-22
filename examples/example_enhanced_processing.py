#!/usr/bin/env python3
"""
Example script demonstrating the enhanced character processing with Gemini AI
"""

import os
import json
from advanced_character_processor import AdvancedCharacterProcessor

def demo_enhanced_processing():
    """
    Demonstrate the enhanced character processing capabilities
    """
    print("🎭 EchoTales Enhanced Character Processing Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("✓ Gemini API key detected")
        print("  Characters will be enhanced with AI-generated insights")
    else:
        print("⚠ No Gemini API key found")
        print("  Set GEMINI_API_KEY environment variable for AI enhancement")
        print("  Processing will continue with standard analysis")
    
    print()
    
    # Initialize processor
    print("🔧 Initializing Enhanced Character Processor...")
    processor = AdvancedCharacterProcessor(gemini_api_key=api_key)
    
    try:
        # Process the book
        print("📚 Processing book characters...")
        output_path, characters = processor.process_book("samples")
        
        print(f"✅ Processing complete!")
        print(f"📁 Output saved to: {output_path}")
        print(f"👥 Total characters processed: {len(characters)}")
        
        # Analyze enhancement coverage
        enhanced_characters = [char for char in characters if 'gemini_enhanced' in char]
        basic_characters = [char for char in characters if 'gemini_enhanced' not in char]
        
        print()
        print("📊 Enhancement Summary:")
        print(f"  🤖 AI-Enhanced: {len(enhanced_characters)} characters")
        print(f"  📝 Basic Analysis: {len(basic_characters)} characters")
        
        # Show detailed examples
        if enhanced_characters:
            print()
            print("🌟 Sample Enhanced Character:")
            char = enhanced_characters[0]
            
            print(f"  Name: {char['name']}")
            print(f"  Importance Score: {char['character_analysis']['importance_score']}")
            print(f"  Gender: {char['gender']}, Age: {char['age_group']}")
            print(f"  Mentions: {char['mention_count']}")
            
            # Show personality
            personality = char['personality']
            print(f"  Personality:")
            print(f"    Openness: {personality['openness']:.2f}")
            print(f"    Conscientiousness: {personality['conscientiousness']:.2f}") 
            print(f"    Extraversion: {personality['extraversion']:.2f}")
            print(f"    Agreeableness: {personality['agreeableness']:.2f}")
            print(f"    Neuroticism: {personality['neuroticism']:.2f}")
            
            # Show enhanced description
            if 'physical_description_enhanced' in char:
                print(f"  📝 Enhanced Physical Description:")
                desc = char['physical_description_enhanced']
                # Show first 200 characters
                preview = desc[:200] + "..." if len(desc) > 200 else desc
                print(f"    {preview}")
            
            # Show Gemini insights
            if 'gemini_enhanced' in char and isinstance(char['gemini_enhanced'], dict):
                gemini_data = char['gemini_enhanced']
                if 'refined_personality' in gemini_data:
                    print(f"  🧠 AI Personality Insight:")
                    insight = str(gemini_data['refined_personality'])[:150] + "..."
                    print(f"    {insight}")
        
        # Show sample quotes
        if characters:
            print()
            print("💬 Sample Character Dialogue:")
            for char in characters[:3]:
                if 'quotes' in char and char['quotes']:
                    quote = char['quotes'][0]
                    quote_text = quote.get('quote_text', '') if isinstance(quote, dict) else str(quote)
                    if quote_text:
                        preview = quote_text[:100] + "..." if len(quote_text) > 100 else quote_text
                        print(f"  {char['name']}: \"{preview}\"")
        
        print()
        print("🎉 Demo complete! Check the output file for full character data.")
        
        return output_path, characters
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

def show_json_structure(output_path):
    """
    Show the structure of the generated JSON file
    """
    print()
    print("📋 Generated JSON Structure:")
    print("-" * 30)
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Root level keys:")
        for key in data.keys():
            print(f"  - {key}")
        
        if 'characters' in data and data['characters']:
            print()
            print("Character entry structure:")
            char = data['characters'][0]
            for key in char.keys():
                value_type = type(char[key]).__name__
                print(f"  - {key} ({value_type})")
        
        if 'processing_info' in data:
            print()
            print("Processing information:")
            info = data['processing_info']
            for key, value in info.items():
                print(f"  - {key}: {value}")
    
    except Exception as e:
        print(f"Error reading JSON structure: {e}")

if __name__ == "__main__":
    try:
        output_path, characters = demo_enhanced_processing()
        show_json_structure(output_path)
    except KeyboardInterrupt:
        print("\n⏹ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        print("Check the README_CHARACTER_ENHANCEMENT.md for troubleshooting tips")