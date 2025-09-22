#!/usr/bin/env python3
"""
BookNLP Output Checker Utility
Checks if BookNLP processing has already been completed for a given book_id
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class BookNLPOutputChecker:
    """Utility class to check if BookNLP output already exists"""
    
    def __init__(self, output_base_dir: Optional[str] = None, generated_base_dir: Optional[str] = None):
        """
        Initialize the checker with output directories
        
        Args:
            output_base_dir: Base directory for BookNLP microservice output
            generated_base_dir: Base directory for SmartBookProcessor generated files
        """
        self.output_base_dir = Path(output_base_dir) if output_base_dir else Path("modelsbooknlp/output")
        self.generated_base_dir = Path(generated_base_dir) if generated_base_dir else Path("modelsbooknlp/generated")
        
    def generate_book_id(self, text: str, title: Optional[str] = None) -> str:
        """Generate consistent book_id from text content"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        if title:
            # Create a more readable book_id with title
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).strip()
            safe_title = safe_title.replace(' ', '_').lower()
            return f"{safe_title}_{text_hash}"
        else:
            return f"book_{text_hash}"
    
    def check_booknlp_output_exists(self, book_id: str) -> bool:
        """
        Check if BookNLP microservice output exists for the given book_id
        
        Args:
            book_id: Unique identifier for the book
            
        Returns:
            bool: True if complete output exists, False otherwise
        """
        book_output_dir = self.output_base_dir / book_id
        
        if not book_output_dir.exists():
            logger.info(f"BookNLP output directory not found: {book_output_dir}")
            return False
        
        # Check for essential BookNLP output files
        essential_files = [
            f"{book_id}.entities",
            f"{book_id}.quotes", 
            f"{book_id}.book"
        ]
        
        missing_files = []
        for filename in essential_files:
            file_path = book_output_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            logger.info(f"Missing BookNLP files for {book_id}: {missing_files}")
            return False
        
        # Check if files have reasonable content (not empty)
        for filename in essential_files:
            file_path = book_output_dir / filename
            if file_path.stat().st_size < 50:  # Minimum reasonable size
                logger.info(f"BookNLP file too small (possibly incomplete): {filename}")
                return False
        
        logger.info(f"✅ Complete BookNLP output found for: {book_id}")
        return True
    
    def check_smart_processor_output_exists(self, book_id: str) -> bool:
        """
        Check if SmartBookProcessor output exists
        
        Args:
            book_id: Unique identifier for the book
            
        Returns:
            bool: True if complete output exists, False otherwise
        """
        # Check for profiles directory
        profiles_dir = self.generated_base_dir / "profiles"
        if not profiles_dir.exists():
            logger.info("SmartBookProcessor profiles directory not found")
            return False
        
        # Check for summary file
        summary_file = profiles_dir / "book_processing_summary.json"
        if not summary_file.exists():
            logger.info("SmartBookProcessor summary file not found")
            return False
        
        # Check for character and scene directories
        portraits_dir = self.generated_base_dir / "character_portraits"
        scenes_dir = self.generated_base_dir / "scene_images"
        
        has_portraits = portraits_dir.exists() and any(portraits_dir.glob("*.jpg"))
        has_scenes = scenes_dir.exists() and any(scenes_dir.glob("*.jpg"))
        
        if not (has_portraits or has_scenes):
            logger.info("SmartBookProcessor generated images not found")
            return False
        
        logger.info(f"✅ Complete SmartBookProcessor output found")
        return True
    
    def get_existing_booknlp_results(self, book_id: str) -> Optional[Dict[str, Any]]:
        """
        Get existing BookNLP results if they exist
        
        Args:
            book_id: Unique identifier for the book
            
        Returns:
            Dict with existing results or None if not found
        """
        if not self.check_booknlp_output_exists(book_id):
            return None
        
        book_output_dir = self.output_base_dir / book_id
        
        try:
            # Get list of generated files
            generated_files = [f.name for f in book_output_dir.iterdir() if f.is_file()]
            
            # Get basic statistics
            entities_count, quotes_count, characters_count = self._extract_statistics(book_output_dir, book_id)
            
            return {
                "status": "success",
                "message": "Using existing BookNLP processing results",
                "book_id": book_id,
                "output_directory": str(book_output_dir),
                "files_generated": generated_files,
                "entities_count": entities_count,
                "quotes_count": quotes_count,
                "characters_count": characters_count,
                "skipped_processing": True
            }
            
        except Exception as e:
            logger.error(f"Error reading existing results: {e}")
            return None
    
    def _extract_statistics(self, output_dir: Path, book_id: str) -> tuple:
        """Extract basic statistics from BookNLP output files"""
        entities_count = None
        quotes_count = None
        characters_count = None
        
        try:
            # Try to read entities file
            entities_file = output_dir / f"{book_id}.entities"
            if entities_file.exists():
                with open(entities_file, 'r', encoding='utf-8') as f:
                    entities_count = sum(1 for line in f) - 1  # Subtract header
            
            # Try to read quotes file
            quotes_file = output_dir / f"{book_id}.quotes"
            if quotes_file.exists():
                with open(quotes_file, 'r', encoding='utf-8') as f:
                    quotes_count = sum(1 for line in f) - 1  # Subtract header
            
            # Try to read book file for character count
            book_file = output_dir / f"{book_id}.book"
            if book_file.exists():
                try:
                    import json
                    with open(book_file, 'r', encoding='utf-8') as f:
                        book_data = json.load(f)
                        if 'characters' in book_data:
                            characters_count = len(book_data['characters'])
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            logger.warning(f"Failed to extract statistics: {e}")
        
        return entities_count, quotes_count, characters_count
    
    def list_existing_books(self) -> List[Dict[str, Any]]:
        """List all books with existing output"""
        existing_books = []
        
        # Check BookNLP outputs
        if self.output_base_dir.exists():
            for book_dir in self.output_base_dir.iterdir():
                if book_dir.is_dir():
                    book_id = book_dir.name
                    if self.check_booknlp_output_exists(book_id):
                        generated_files = [f.name for f in book_dir.iterdir() if f.is_file()]
                        entities_count, quotes_count, characters_count = self._extract_statistics(book_dir, book_id)
                        
                        existing_books.append({
                            "book_id": book_id,
                            "type": "booknlp",
                            "output_directory": str(book_dir),
                            "files_generated": generated_files,
                            "entities_count": entities_count,
                            "quotes_count": quotes_count,
                            "characters_count": characters_count
                        })
        
        return existing_books
    
    def clean_incomplete_outputs(self, book_id: Optional[str] = None) -> int:
        """
        Clean up incomplete or corrupted output directories
        
        Args:
            book_id: Specific book to clean, or None to clean all incomplete outputs
            
        Returns:
            int: Number of directories cleaned up
        """
        cleaned_count = 0
        
        if book_id:
            # Clean specific book
            book_output_dir = self.output_base_dir / book_id
            if book_output_dir.exists() and not self.check_booknlp_output_exists(book_id):
                import shutil
                shutil.rmtree(book_output_dir)
                cleaned_count = 1
                logger.info(f"Cleaned incomplete output for: {book_id}")
        else:
            # Clean all incomplete outputs
            if self.output_base_dir.exists():
                for book_dir in self.output_base_dir.iterdir():
                    if book_dir.is_dir():
                        book_id_to_check = book_dir.name
                        if not self.check_booknlp_output_exists(book_id_to_check):
                            import shutil
                            shutil.rmtree(book_dir)
                            cleaned_count += 1
                            logger.info(f"Cleaned incomplete output for: {book_id_to_check}")
        
        return cleaned_count


def main():
    """Test the BookNLPOutputChecker"""
    checker = BookNLPOutputChecker()
    
    # List existing books
    existing = checker.list_existing_books()
    print(f"Found {len(existing)} books with existing output:")
    for book in existing:
        print(f"  - {book['book_id']}: {len(book['files_generated'])} files")
    
    # Test book ID generation
    sample_text = "This is a sample book text for testing."
    book_id = checker.generate_book_id(sample_text, "Sample Book")
    print(f"\nGenerated book_id: {book_id}")
    print(f"Output exists: {checker.check_booknlp_output_exists(book_id)}")


if __name__ == "__main__":
    main()