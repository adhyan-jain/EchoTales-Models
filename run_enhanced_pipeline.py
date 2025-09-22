#!/usr/bin/env python3
"""
Enhanced Pipeline Runner with Gemini API Integration
Demonstrates the enhanced character processing and dialogue attribution
"""

import os
import json
import logging
from pathlib import Path
from advanced_character_processor import AdvancedCharacterProcessor
from dialogue_processor import FixedDialogueProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the enhanced pipeline with Gemini API integration"""
    
    # Check for Gemini API key
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Using fallback methods.")
        logger.info("To use Gemini API features, set your API key:")
        logger.info("export GEMINI_API_KEY='your_api_key_here'")
    else:
        logger.info("Gemini API key found. Enhanced features will be enabled.")
    
    book_id = "samples"
    
    try:
        # Step 1: Enhanced Character Processing with Gemini API
        logger.info("=" * 60)
        logger.info("STEP 1: Enhanced Character Processing with Gemini API")
        logger.info("=" * 60)
        
        character_processor = AdvancedCharacterProcessor(gemini_api_key=gemini_api_key)
        character_output_path, characters = character_processor.process_book(book_id)
        
        logger.info(f"✅ Character processing complete!")
        logger.info(f"📁 Characters saved to: {character_output_path}")
        logger.info(f"👥 Total characters: {len(characters)}")
        
        # Show enhanced character analysis
        logger.info(f"\n🔍 Enhanced Character Analysis:")
        for i, char in enumerate(characters[:5], 1):
            logger.info(f"{i}. {char['name']} (ID: {char['character_id']})")
            logger.info(f"   Gender: {char['gender']} | Age: {char['age_group']}")
            logger.info(f"   Personality: O={char['personality']['openness']:.2f}, C={char['personality']['conscientiousness']:.2f}, E={char['personality']['extraversion']:.2f}, A={char['personality']['agreeableness']:.2f}, N={char['personality']['neuroticism']:.2f}")
            logger.info(f"   Quotes: {len(char['representative_quotes'])} representative quotes")
            logger.info(f"   Gemini Analyzed: {char['character_analysis']['gemini_analyzed']}")
            logger.info()
        
        # Step 2: Enhanced Dialogue Processing with Gemini API
        logger.info("=" * 60)
        logger.info("STEP 2: Enhanced Dialogue Processing with Gemini API")
        logger.info("=" * 60)
        
        dialogue_processor = FixedDialogueProcessor(gemini_api_key=gemini_api_key)
        dialogue_summary, dialogue_files = dialogue_processor.process_book_dialogue(book_id)
        
        logger.info(f"✅ Dialogue processing complete!")
        logger.info(f"📁 Total chapter files: {len(dialogue_files)}")
        logger.info(f"📊 Summary:")
        logger.info(f"   Chapters: {dialogue_summary['total_chapters']}")
        logger.info(f"   Total lines: {dialogue_summary['total_dialogue_lines']}")
        logger.info(f"   Characters: {dialogue_summary['unique_characters']}")
        logger.info(f"   Newly processed: {dialogue_summary.get('newly_processed_chapters', 0)}")
        logger.info(f"   Already existed: {dialogue_summary.get('existing_chapters', 0)}")
        
        # Show sample dialogue with enhanced attribution
        logger.info(f"\n💬 Sample Dialogue with Enhanced Attribution:")
        sample_files = [f for f in dialogue_files if Path(f).exists()][:2]  # First 2 files
        for i, filepath in enumerate(sample_files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    chapter_data = json.load(f)
                    logger.info(f"\n📖 Chapter {chapter_data['chapter']} ({Path(filepath).name}):")
                    logger.info(f"   Lines: {chapter_data['total_lines']}")
                    logger.info(f"   Characters: {len(chapter_data['characters_in_chapter'])}")
                    
                    # Show first 3 lines with character attribution
                    for line in chapter_data['lines'][:3]:
                        logger.info(f"   {line['character_name']}: \"{line['text'][:60]}...\" ({line['emotion']})")
            except Exception as e:
                logger.warning(f"   Could not load chapter data: {e}")
        
        # Step 3: Summary Report
        logger.info("=" * 60)
        logger.info("ENHANCEMENT SUMMARY")
        logger.info("=" * 60)
        logger.info("✅ Added 6 representative quotes per character")
        logger.info("✅ Integrated Gemini API for character analysis")
        logger.info("✅ Enhanced pronoun character removal and quote reassignment")
        logger.info("✅ Improved quote attribution using AI")
        logger.info("✅ Gemini-powered gender and OCEAN trait classification")
        logger.info("✅ Better character naming (avoiding generic Character_1, etc.)")
        
        if gemini_api_key:
            logger.info("🤖 Gemini API features: ENABLED")
        else:
            logger.info("⚠️  Gemini API features: DISABLED (no API key)")
        
        logger.info(f"\n🎉 Enhanced pipeline complete! Check the 'characters/' and 'dialogue/' directories for results.")
        
    except Exception as e:
        logger.error(f"Enhanced pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
