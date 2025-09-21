#!/usr/bin/env python3
"""
Pipeline to generate chapter-wise dialogue JSONs using
AdvancedCharacterProcessor for character info.
"""

import logging
from pathlib import Path
import os

from advanced_character_processor import AdvancedCharacterProcessor
from dialogue_processor import FixedDialogueProcessor

# --------------------------
# ⚡ Hardcoded environment
# --------------------------
os.environ["GEMINI_API_KEY"] = "AIzaSyD23o0K1sI4lghuNJhnEnUMu_B-jaUf9i4"
os.environ["GEMINI_API_URL"] = "https://gemini.example.com/v1/analyze"  # Replace with actual endpoint

# --------------------------
# Logger setup
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------
# Main function
# --------------------------
def main(book_id: str):
    # 1️⃣ Generate enhanced character info
    char_processor = AdvancedCharacterProcessor()
    logger.info(f"Generating enhanced character entries for book '{book_id}'...")

    # Use method compatible with pipeline
    char_json_path, character_list = char_processor.process_book(book_id)
    logger.info(f"✅ Enhanced characters saved to: {char_json_path}")

    # 2️⃣ Save JSON for dialogue processor
    characters_file_for_dialogue = Path("characters/for_dialogue.json")
    characters_file_for_dialogue.parent.mkdir(parents=True, exist_ok=True)
    characters_file_for_dialogue.write_text(char_json_path.read_text(encoding='utf-8'), encoding='utf-8')
    logger.info(f"Saved character info for dialogue processing: {characters_file_for_dialogue}")

    # 3️⃣ Generate chapter-wise dialogue JSONs
    dialogue_processor = FixedDialogueProcessor()
    logger.info(f"Processing chapter-wise dialogue for book '{book_id}'...")
    summary, saved_files = dialogue_processor.process_book_dialogue(
        book_id,
        characters_file=str(characters_file_for_dialogue)
    )

    # 4️⃣ Log summary
    logger.info(f"\n✅ Chapter-wise dialogue processing complete!")
    logger.info(f"📁 Total chapter files: {len(saved_files)}")
    logger.info(f"📊 Summary: {summary}")

    # Preview first few chapters
    for filepath in saved_files[:3]:
        logger.info(f"First chapters preview: {filepath}")

if __name__ == "__main__":
    main(book_id="samples")
