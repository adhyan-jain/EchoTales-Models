#!/usr/bin/env python3
"""
Copy Sample Data Script
Copy sample BookNLP output from EchoTales-Models to test our processing pipeline
"""

import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def copy_sample_booknlp_data():
    """Copy sample BookNLP data for testing"""
    
    # Source directory (EchoTales-Models)
    source_dir = Path("C:/Users/Adhyan/OneDrive/Desktop/EchoTales-Models/modelsbooknlp/output")
    
    # Destination directory (EchoTales-Enhanced)
    dest_dir = Path("C:/Users/Adhyan/OneDrive/Desktop/EchoTales-Enhanced/data/processed/booknlp")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Find a sample book ID from the source
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return
    
    # Look for .entities files to determine book IDs
    entities_files = list(source_dir.glob("*.entities"))
    
    if not entities_files:
        logger.error("No .entities files found in source directory")
        return
    
    # Use the first book ID found
    sample_file = entities_files[0]
    book_id = sample_file.stem
    
    logger.info(f"Found sample book_id: {book_id}")
    
    # Copy the three main files we need
    files_to_copy = [
        f"{book_id}.entities",
        f"{book_id}.tokens", 
        f"{book_id}.quotes"
    ]
    
    for filename in files_to_copy:
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        
        if source_file.exists():
            logger.info(f"Copying {filename}...")
            shutil.copy2(source_file, dest_file)
            logger.info(f"✓ Copied to {dest_file}")
        else:
            logger.warning(f"Source file not found: {source_file}")
    
    logger.info(f"Sample data copy completed. Book ID: {book_id}")
    return book_id

def create_sample_text_file(book_id: str):
    """Create a sample text file for the book"""
    
    dest_dir = Path("C:/Users/Adhyan/OneDrive/Desktop/EchoTales-Enhanced/data/raw/books")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    sample_text = """[CHAPTER_START_1]
Chapter 1: The Beginning

This is the beginning of our test story. The main character, Klein, wakes up feeling confused.

"Where am I?" Klein asked himself, looking around the unfamiliar room.

The narrator describes the scene as Klein tries to make sense of his surroundings.

[CHAPTER_START_2]
Chapter 2: Discovery

Klein discovered something important about his situation. He was no longer in his original world.

"This is impossible," he said, staring at his reflection.

The story continues with more dialogue and narrative sections.
"""
    
    sample_file = dest_dir / f"{book_id}.txt"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    logger.info(f"Created sample text file: {sample_file}")

def main():
    """Main function"""
    logger.info("Starting sample data copy...")
    
    try:
        book_id = copy_sample_booknlp_data()
        if book_id:
            create_sample_text_file(book_id)
            logger.info("✅ Sample data setup completed successfully!")
            logger.info(f"You can now test with: python scripts/processing/enhanced_character_processor.py --book-id {book_id}")
        else:
            logger.error("Failed to copy sample data")
            
    except Exception as e:
        logger.error(f"Error during sample data copy: {e}")

if __name__ == "__main__":
    main()