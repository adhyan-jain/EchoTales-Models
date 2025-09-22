#!/usr/bin/env python3
"""
Interactive Book and Chapter Processor for EchoTales
Allows user to select books and chapters for comprehensive processing
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import our processors
from generate_comprehensive_json import ComprehensiveJSONGenerator
from chapter_segment_processor import ChapterSegmentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BookLibrary:
    """Manages available books and their metadata"""
    
    def __init__(self):
        self.books_dir = Path("data/raw/books")
        self.books_dir.mkdir(parents=True, exist_ok=True)
        self.available_books = self._scan_available_books()
    
    def _scan_available_books(self) -> Dict[str, Dict[str, Any]]:
        """Scan for available books in the library"""
        
        books = {}
        
        # Look for text files in books directory
        if self.books_dir.exists():
            for book_file in self.books_dir.glob("*.txt"):
                book_info = self._analyze_book_file(book_file)
                if book_info:
                    books[book_info["book_id"]] = book_info
        
        # Add sample book if no books found
        if not books:
            books["sample"] = {
                "book_id": "sample",
                "title": "The Marketplace Chronicles",
                "author": "Demo Author",
                "file_path": None,  # Will use built-in sample
                "estimated_chapters": 2,
                "file_size": "~1KB (built-in sample)",
                "description": "A sample story about Emma and Thomas in a medieval marketplace"
            }
        
        return books
    
    def _analyze_book_file(self, book_file: Path) -> Optional[Dict[str, Any]]:
        """Analyze a book file to extract metadata"""
        
        try:
            # Read first few lines to get metadata
            with open(book_file, 'r', encoding='utf-8') as f:
                content_sample = f.read(2000)  # First 2KB
            
            # Create book_id from filename
            book_id = book_file.stem.lower().replace(' ', '_').replace('-', '_')
            
            # Extract title from filename
            title = book_file.stem.replace('_', ' ').title()
            
            # Estimate chapters by looking for chapter markers
            chapter_count = len(re.findall(r'\[CHAPTER_START_\d+\]', content_sample))
            if chapter_count == 0:
                # Fallback: look for "Chapter" mentions
                chapter_count = len(re.findall(r'Chapter \d+', content_sample[:5000]))
            if chapter_count == 0:
                chapter_count = 1  # Default to single chapter
            
            # Get file size
            file_size = f"{book_file.stat().st_size // 1024}KB"
            if book_file.stat().st_size > 1024 * 1024:
                file_size = f"{book_file.stat().st_size // (1024 * 1024)}MB"
            
            # Try to detect author from content
            author = "Unknown Author"
            author_patterns = [
                r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                r'Author:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)'
            ]
            for pattern in author_patterns:
                match = re.search(pattern, content_sample)
                if match:
                    author = match.group(1)
                    break
            
            return {
                "book_id": book_id,
                "title": title,
                "author": author,
                "file_path": str(book_file),
                "estimated_chapters": max(1, chapter_count),
                "file_size": file_size,
                "description": f"Novel: {title} by {author}"
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze {book_file}: {e}")
            return None
    
    def get_available_books(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available books"""
        return self.available_books
    
    def get_book_content(self, book_id: str) -> Optional[str]:
        """Get the content of a specific book"""
        
        if book_id not in self.available_books:
            return None
        
        book_info = self.available_books[book_id]
        
        # Handle built-in sample
        if book_info["file_path"] is None and book_id == "sample":
            return self._get_sample_content()
        
        # Read from file
        try:
            with open(book_info["file_path"], 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Could not read book {book_id}: {e}")
            return None
    
    def _get_sample_content(self) -> str:
        """Get built-in sample content"""
        return '''[CHAPTER_START_1]
Emma walked through the busy marketplace in London, her basket clutched tightly in her hand. The morning sun cast long shadows between the wooden stalls.

"Fresh apples! Best in all of England!" called out Thomas, the fruit vendor, waving to catch her attention.

Emma smiled and approached his stall. "Good morning, Thomas. How much for a dozen?"

"For you, my dear Emma, just two shillings," Thomas replied with a grin. "Your father was always good to me."

"Thank you," Emma said, placing the coins in his weathered palm. "Has anyone seen my brother James today? He was supposed to meet me here."

Thomas shook his head. "No, but Mrs. Patterson mentioned she saw him near the cathedral this morning."

[CHAPTER_START_2]
Near the cathedral, James was meeting with Father McKenzie about the village's ancient history. The old priest had many stories to tell about the war.

"Tell me more about our grandfather," James said eagerly.

Father McKenzie nodded. "He served with honor in France. Your family has a proud heritage."

The church bells rang across the town square as they continued their conversation.

Emma found them there an hour later, deep in discussion about family secrets that would change everything she thought she knew about their past.'''


class InteractiveProcessor:
    """Main interactive processor for books and chapters"""
    
    def __init__(self):
        self.library = BookLibrary()
        self.json_generator = ComprehensiveJSONGenerator()
        self.segment_processor = ChapterSegmentProcessor()
        
        print("🎬 EchoTales Interactive Book Processor")
        print("=" * 50)
        print("📚 Generates comprehensive JSON files and scene images")
        print("🖼️  Creates character profiles with animations")
        print("🎭 Processes chapters into segments with backgrounds")
        print()
    
    def run(self):
        """Main interactive loop"""
        
        while True:
            try:
                choice = self._show_main_menu()
                
                if choice == "1":
                    self._process_full_book()
                elif choice == "2":
                    self._process_chapter_segments()
                elif choice == "3":
                    self._view_available_books()
                elif choice == "4":
                    self._view_generated_files()
                elif choice == "0":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")
    
    def _show_main_menu(self) -> str:
        """Show main menu and get user choice"""
        
        print("\n" + "=" * 50)
        print("🎯 MAIN MENU")
        print("=" * 50)
        print("1. 📖 Generate Comprehensive JSON (Full Book)")
        print("2. 🎬 Process Chapter Segments (Scene Images)")
        print("3. 📚 View Available Books")
        print("4. 📁 View Generated Files")
        print("0. 🚪 Exit")
        print()
        
        return input("Choose an option (0-4): ").strip()
    
    def _select_book(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Let user select a book"""
        
        books = self.library.get_available_books()
        
        if not books:
            print("❌ No books available!")
            return None
        
        print("\n📚 Available Books:")
        print("-" * 30)
        
        book_list = list(books.items())
        for i, (book_id, book_info) in enumerate(book_list, 1):
            print(f"{i}. {book_info['title']}")
            print(f"   Author: {book_info['author']}")
            print(f"   Chapters: ~{book_info['estimated_chapters']}")
            print(f"   Size: {book_info['file_size']}")
            print()
        
        try:
            choice = input(f"Select book (1-{len(book_list)}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(book_list):
                selected_book = book_list[choice_num - 1]
                print(f"✅ Selected: {selected_book[1]['title']}")
                return selected_book
            else:
                print("❌ Invalid selection")
                return None
                
        except ValueError:
            print("❌ Please enter a valid number")
            return None
    
    def _select_chapter(self, book_content: str, book_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Let user select a chapter or process entire book"""
        
        # Split content to find chapters
        chapters = self._split_into_chapters(book_content)
        
        print(f"\n📖 Chapters in '{book_info['title']}':")
        print("-" * 30)
        print("0. 📚 Process Entire Book")
        
        for i, chapter in enumerate(chapters, 1):
            word_count = len(chapter['content'].split())
            print(f"{i}. {chapter['title']} ({word_count} words)")
        
        try:
            choice = input(f"Select chapter (0-{len(chapters)}): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                return {"process_all": True, "chapters": chapters}
            elif 1 <= choice_num <= len(chapters):
                selected_chapter = chapters[choice_num - 1]
                print(f"✅ Selected: {selected_chapter['title']}")
                return {"process_all": False, "chapter": selected_chapter}
            else:
                print("❌ Invalid selection")
                return None
                
        except ValueError:
            print("❌ Please enter a valid number")
            return None
    
    def _split_into_chapters(self, book_content: str) -> List[Dict[str, Any]]:
        """Split book content into chapters"""
        
        chapter_pattern = r'\[CHAPTER_START_(\d+)\]'
        chapters = []
        
        # Split by chapter markers
        parts = re.split(chapter_pattern, book_content)
        
        if len(parts) <= 1:
            # No chapter markers, treat as single chapter
            chapters.append({
                "chapter_number": 1,
                "title": "Chapter 1",
                "content": book_content.strip(),
                "word_count": len(book_content.split())
            })
        else:
            # Process marked chapters
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    page_number = int(parts[i])
                    content = parts[i + 1].strip()
                    
                    if content:
                        chapters.append({
                            "chapter_number": (i // 2) + 1,
                            "title": f"Chapter {(i // 2) + 1}",
                            "content": content,
                            "word_count": len(content.split()),
                            "page_number": page_number
                        })
        
        return chapters
    
    def _process_full_book(self):
        """Process a full book with comprehensive JSON generation"""
        
        print("\n🎯 COMPREHENSIVE JSON GENERATION")
        print("=" * 40)
        
        # Select book
        book_selection = self._select_book()
        if not book_selection:
            return
        
        book_id, book_info = book_selection
        
        # Get book content
        book_content = self.library.get_book_content(book_id)
        if not book_content:
            print(f"❌ Could not read book content for {book_info['title']}")
            return
        
        print(f"\n📖 Processing: {book_info['title']}")
        print(f"📊 Content length: {len(book_content)} characters")
        print("🔄 This may take several minutes...")
        print()
        
        # Process with comprehensive JSON generator
        try:
            result = self.json_generator.process_novel(
                novel_text=book_content,
                title=book_info['title'],
                author=book_info['author']
            )
            
            if result["success"]:
                print("✅ Comprehensive JSON generation complete!")
                self._show_processing_results(result)
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Processing error: {e}")
    
    def _process_chapter_segments(self):
        """Process chapter segments with scene images"""
        
        print("\n🎬 CHAPTER SEGMENT PROCESSING")
        print("=" * 40)
        
        # Select book
        book_selection = self._select_book()
        if not book_selection:
            return
        
        book_id, book_info = book_selection
        
        # Get book content
        book_content = self.library.get_book_content(book_id)
        if not book_content:
            print(f"❌ Could not read book content for {book_info['title']}")
            return
        
        # Select chapter
        chapter_selection = self._select_chapter(book_content, book_info)
        if not chapter_selection:
            return
        
        try:
            if chapter_selection["process_all"]:
                # Process all chapters
                chapters = chapter_selection["chapters"]
                print(f"\n🔄 Processing {len(chapters)} chapters...")
                
                for chapter in chapters:
                    print(f"\n📖 Processing {chapter['title']}...")
                    result = self.segment_processor.process_chapter(
                        chapter_text=chapter['content'],
                        chapter_info=chapter
                    )
                    
                    if result:
                        print(f"✅ {chapter['title']} processed successfully!")
                        summary = result["processing_summary"]
                        print(f"   Segments: {summary['total_segments']}")
                        print(f"   Images: {summary['total_scenes_generated']}")
                        print(f"   Characters: {', '.join(summary['unique_characters'][:3])}...")
                    
            else:
                # Process single chapter
                chapter = chapter_selection["chapter"]
                print(f"\n🔄 Processing {chapter['title']}...")
                
                result = self.segment_processor.process_chapter(
                    chapter_text=chapter['content'],
                    chapter_info=chapter
                )
                
                if result:
                    print("✅ Chapter segment processing complete!")
                    self._show_segment_results(result)
                else:
                    print("❌ Chapter processing failed!")
                    
        except Exception as e:
            print(f"❌ Processing error: {e}")
    
    def _show_processing_results(self, result: Dict[str, Any]):
        """Show comprehensive processing results"""
        
        print("\n📊 PROCESSING RESULTS:")
        print("-" * 30)
        
        summary = result["novel_info"]
        print(f"📚 Title: {summary['title']}")
        print(f"✍️  Author: {summary['author']}")
        print(f"📄 Chapters: {summary['total_chapters']}")
        print(f"📝 Words: {summary['total_words']}")
        print(f"🤖 BookNLP: {'Used' if summary.get('booknlp_used', False) else 'Fallback analysis'}")
        
        # Count generated files
        files = result.get("generated_files", {})
        total_chars = len(files.get("characters", []))
        total_chapters = len(files.get("chapters", []))
        total_backgrounds = len(files.get("backgrounds", []))
        
        print(f"\n📁 Generated Files:")
        print(f"  👥 Characters: {total_chars} profiles with images & animations")
        print(f"  📖 Chapters: {total_chapters} chapter JSON files")
        print(f"  🏞️  Backgrounds: {total_backgrounds} setting images")
        
        if "master_index" in files:
            print(f"\n🗂️  Master index: {files['master_index']}")
    
    def _show_segment_results(self, result: Dict[str, Any]):
        """Show segment processing results"""
        
        print("\n📊 SEGMENT PROCESSING RESULTS:")
        print("-" * 30)
        
        summary = result["processing_summary"]
        print(f"📄 Total segments: {summary['total_segments']}")
        print(f"🖼️  Scene images: {summary['total_scenes_generated']}")
        print(f"👥 Characters: {', '.join(summary['unique_characters'])}")
        print(f"🏞️  Settings: {', '.join(summary['unique_settings'])}")
        
        # Show sample segment
        if result.get("segments"):
            segment = result["segments"][0]
            print(f"\n🎬 Sample Segment ({segment['segment_id']}):")
            print(f"  Characters: {', '.join(segment['characters_present'][:3])}")
            print(f"  Setting: {segment['detected_setting']}")
            print(f"  Mood: {segment['scene_mood']}")
            print(f"  Image: {Path(segment['scene_image']).name}")
    
    def _view_available_books(self):
        """Show detailed view of available books"""
        
        print("\n📚 AVAILABLE BOOKS")
        print("=" * 40)
        
        books = self.library.get_available_books()
        
        if not books:
            print("❌ No books found!")
            print("📁 Add book files to: data/raw/books/")
            return
        
        for book_id, book_info in books.items():
            print(f"📖 {book_info['title']}")
            print(f"   ID: {book_id}")
            print(f"   Author: {book_info['author']}")
            print(f"   Chapters: ~{book_info['estimated_chapters']}")
            print(f"   Size: {book_info['file_size']}")
            print(f"   Description: {book_info['description']}")
            if book_info.get('file_path'):
                print(f"   File: {Path(book_info['file_path']).name}")
            print()
    
    def _view_generated_files(self):
        """Show generated files"""
        
        print("\n📁 GENERATED FILES")
        print("=" * 40)
        
        data_dir = Path("data/generated")
        
        if not data_dir.exists():
            print("❌ No generated files found!")
            print("   Run processing first to generate files.")
            return
        
        # Show different types of generated content
        sections = {
            "json": "📋 JSON Files (Character/Chapter data)",
            "images": "🖼️  Generated Images",
            "scenes": "🎬 Scene Images", 
            "segments": "📖 Chapter Segments",
            "animations": "✨ Animation Files"
        }
        
        for folder_name, description in sections.items():
            folder_path = data_dir / folder_name
            if folder_path.exists():
                files = list(folder_path.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                
                if file_count > 0:
                    print(f"{description}: {file_count} files")
                    print(f"   📁 {folder_path}")
                    
                    # Show sample files
                    sample_files = [f for f in files if f.is_file()][:3]
                    for sample_file in sample_files:
                        print(f"      📄 {sample_file.name}")
                    
                    if len(sample_files) < file_count:
                        print(f"      ... and {file_count - len(sample_files)} more")
                    print()
        
        print(f"🗂️  Total generated directory: {data_dir}")


def main():
    """Main function"""
    try:
        processor = InteractiveProcessor()
        processor.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main()