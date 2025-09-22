#!/usr/bin/env python3
"""
Advanced Pipeline Runner
Runs the advanced character processing system with dialogue processing
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
import traceback

# Import our advanced processors
from advanced_character_processor import AdvancedCharacterProcessor
from dialogue_processor import FixedDialogueProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPipelineRunner:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        
        # Updated paths
        self.output_dir = self.base_dir / "modelsbooknlp" / "output"
        self.characters_dir = self.base_dir / "characters"
        self.dialogue_dir = self.base_dir / "dialogue"
        
        # Create directories if they don't exist
        self.characters_dir.mkdir(exist_ok=True)
        self.dialogue_dir.mkdir(exist_ok=True)
        
        # Initialize processors
        self.character_processor = AdvancedCharacterProcessor(
            output_dir=str(self.output_dir),
            characters_dir=str(self.characters_dir)
        )
        self.dialogue_processor = FixedDialogueProcessor(
            output_dir=str(self.output_dir),
            dialogue_dir=str(self.dialogue_dir)
        )
        
        self.results = {}

    def discover_book_ids(self) -> List[str]:
        """Discover all book IDs from BookNLP output directory"""
        if not self.output_dir.exists():
            logger.warning(f"BookNLP output directory not found: {self.output_dir}")
            return []
        
        book_ids = set()
        for file_path in self.output_dir.glob("*.entities"):
            book_id = file_path.stem
            book_ids.add(book_id)
        
        return sorted(list(book_ids))

    def process_single_book(self, book_id: str) -> Dict[str, Any]:
        """Process a single book through the advanced pipeline"""
        logger.info(f"Starting advanced processing for: {book_id}")
        start_time = time.time()
        
        result = {
            'book_id': book_id,
            'success': False,
            'processing_time': 0,
            'character_count': 0,
            'dialogue_lines': 0,
            'chapters_processed': 0,
            'errors': []
        }
        
        try:
            # Step 1: Advanced Character Processing
            logger.info(f"Step 1: Advanced character processing...")
            char_start = time.time()
            
            char_output_path, characters = self.character_processor.process_book(book_id)
            result['character_count'] = len(characters)
            result['character_processing_time'] = time.time() - char_start
            
            logger.info(f"Character processing complete: {len(characters)} characters")
            
            # Step 2: Dialogue Processing
            logger.info(f"Step 2: Dialogue processing...")
            dialogue_start = time.time()
            
            dialogue_summary, dialogue_files = self.dialogue_processor.process_book_dialogue(
                book_id, char_output_path
            )
            
            result['dialogue_lines'] = dialogue_summary.get('total_dialogue_lines', 0)
            result['chapters_processed'] = dialogue_summary.get('total_chapters', 0)
            result['dialogue_processing_time'] = time.time() - dialogue_start
            
            logger.info(f"Dialogue processing complete: {result['dialogue_lines']} lines in {result['chapters_processed']} chapters")
            
            # Step 3: Generate comprehensive report
            result['success'] = True
            result['character_file'] = str(char_output_path)
            result['dialogue_files'] = dialogue_files
            result['summary'] = dialogue_summary
            
            # Calculate total processing time
            result['processing_time'] = time.time() - start_time
            
            logger.info(f"Advanced processing complete for {book_id} in {result['processing_time']:.2f}s")
            
        except Exception as e:
            error_msg = f"Error processing {book_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            result['errors'].append(error_msg)
            result['processing_time'] = time.time() - start_time
        
        return result

    def process_multiple_books(self, book_ids: List[str]) -> Dict[str, Any]:
        """Process multiple books"""
        logger.info(f"Processing {len(book_ids)} books...")
        
        all_results = {}
        total_start_time = time.time()
        
        for i, book_id in enumerate(book_ids, 1):
            logger.info(f"\nProcessing book {i}/{len(book_ids)}: {book_id}")
            result = self.process_single_book(book_id)
            all_results[book_id] = result
            
            # Brief pause between books
            time.sleep(0.5)
        
        total_time = time.time() - total_start_time
        
        # Generate summary
        successful_books = [bid for bid, result in all_results.items() if result['success']]
        failed_books = [bid for bid, result in all_results.items() if not result['success']]
        
        total_characters = sum(result['character_count'] for result in all_results.values())
        total_dialogue = sum(result['dialogue_lines'] for result in all_results.values())
        total_chapters = sum(result['chapters_processed'] for result in all_results.values())
        
        summary = {
            'total_books_processed': len(book_ids),
            'successful_books': len(successful_books),
            'failed_books': len(failed_books),
            'total_processing_time': total_time,
            'total_characters': total_characters,
            'total_dialogue_lines': total_dialogue,
            'total_chapters': total_chapters,
            'successful_book_ids': successful_books,
            'failed_book_ids': failed_books,
            'book_results': all_results
        }
        
        logger.info(f"\nPROCESSING SUMMARY")
        logger.info(f"=" * 50)
        logger.info(f"Total books: {len(book_ids)}")
        logger.info(f"Successful: {len(successful_books)}")
        logger.info(f"Failed: {len(failed_books)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Total characters: {total_characters}")
        logger.info(f"Total dialogue lines: {total_dialogue}")
        logger.info(f"Total chapters: {total_chapters}")
        
        if failed_books:
            logger.warning(f"Failed books: {', '.join(failed_books)}")
        
        return summary

    def save_processing_report(self, summary: Dict[str, Any], filename: str = None):
        """Save processing report to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_processing_report_{timestamp}.json"
        
        report_path = self.base_dir / "batch_results" / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Processing report saved to: {report_path}")
        return str(report_path)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Character Processing Pipeline")
    parser.add_argument("--book-ids", nargs="+", help="Specific book IDs to process")
    parser.add_argument("--auto-discover", action="store_true", help="Auto-discover all books")
    parser.add_argument("--report", action="store_true", help="Generate processing report")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AdvancedPipelineRunner()
    
    # Determine which books to process
    if args.book_ids:
        book_ids = args.book_ids
        logger.info(f"Processing specified books: {book_ids}")
    elif args.auto_discover:
        book_ids = pipeline.discover_book_ids()
        if not book_ids:
            logger.error("No books found for auto-discovery")
            sys.exit(1)
        logger.info(f"Auto-discovered books: {book_ids}")
    else:
        # Default to samples
        book_ids = ["samples"]
        logger.info(f"Processing default book: {book_ids}")
    
    try:
        # Process books
        if len(book_ids) == 1:
            result = pipeline.process_single_book(book_ids[0])
            summary = {
                'total_books_processed': 1,
                'successful_books': 1 if result['success'] else 0,
                'failed_books': 0 if result['success'] else 1,
                'book_results': {book_ids[0]: result}
            }
        else:
            summary = pipeline.process_multiple_books(book_ids)
        
        # Save report if requested
        if args.report:
            report_path = pipeline.save_processing_report(summary)
            print(f"\n📄 Report saved to: {report_path}")
        
        # Exit with appropriate code
        if summary['failed_books'] > 0:
            logger.warning(f"Processing completed with {summary['failed_books']} failures")
            sys.exit(1)
        else:
            logger.info("All processing completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
