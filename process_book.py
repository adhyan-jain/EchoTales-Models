#!/usr/bin/env python3
"""
EchoTales Smart Book Processor
Clean interface for processing books with real character detection and quality images
"""

import os
import sys
from pathlib import Path
from smart_book_processor import SmartBookProcessor
from utils.booknlp_output_checker import BookNLPOutputChecker

def main():
    """Main function"""
    
    print("🎬 EchoTales Smart Book Processor")
    print("=" * 50)
    print("✨ Intelligently detects REAL characters (not random words)")
    print("🎨 Generates high-quality character portraits")
    print("🎬 Creates beautiful scene images")
    print("📋 Produces clean, focused output")
    print("⏩ Automatically skips books that have already been processed")
    print()
    
    # Initialize output checker
    output_checker = BookNLPOutputChecker()
    
    # Check for books
    books_dir = Path("modelsbooknlp")
    if not books_dir.exists():
        print("❌ No modelsbooknlp directory found!")
        print("📁 Please ensure modelsbooknlp directory exists with book files")
        return
    
    # Find available books
    book_files = list(books_dir.glob("*.txt"))
    
    if not book_files:
        print("❌ No book files found!")
        print("📁 Please add .txt book files to 'modelsbooknlp/' directory")
        return
    
    # Check for existing processed books
    existing_books = output_checker.list_existing_books()
    if existing_books:
        print(f"📚 {len(existing_books)} books already processed:")
        for book in existing_books[:3]:  # Show first 3
            entities = book.get('entities_count', 0) or 0
            characters = book.get('characters_count', 0) or 0
            print(f"  ✓ {book['book_id']}: {entities} entities, {characters} characters")
        if len(existing_books) > 3:
            print(f"  ... and {len(existing_books) - 3} more")
        print()
    
    # Show available books
    print("📚 Available Books:")
    print("-" * 30)
    
    for i, book_file in enumerate(book_files, 1):
        file_size = book_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        title = book_file.stem.replace('_', ' ').title()
        
        # Check if this book has been processed
        with open(book_file, 'r', encoding='utf-8') as f:
            sample_text = f.read(1000)  # Read first 1000 chars for ID generation
        book_id = output_checker.generate_book_id(sample_text, title)
        already_processed = output_checker.check_booknlp_output_exists(book_id) or output_checker.check_smart_processor_output_exists(book_id)
        status = "✅ PROCESSED" if already_processed else "🆕 NEW"
        
        print(f"{i}. {title} [{status}]")
        print(f"   File: {book_file.name}")
        print(f"   Size: {size_mb:.1f}MB")
        if already_processed:
            print(f"   📄 Book ID: {book_id}")
        print()
    
    # Let user choose
    try:
        choice = input(f"Select book (1-{len(book_files)}): ").strip()
        choice_num = int(choice)
        
        if not (1 <= choice_num <= len(book_files)):
            print("❌ Invalid selection")
            return
        
        selected_book = book_files[choice_num - 1]
        
    except ValueError:
        print("❌ Please enter a valid number")
        return
    
    # Read the book
    print(f"\n📖 Loading: {selected_book.name}")
    
    try:
        with open(selected_book, 'r', encoding='utf-8') as f:
            book_content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Extract title and author from filename/content
    title = selected_book.stem.replace('_', ' ').title()
    author = "Unknown Author"
    
    # Try to detect author from content
    content_lines = book_content.split('\n')[:10]  # First 10 lines
    for line in content_lines:
        line_lower = line.lower()
        if 'by ' in line_lower or 'author:' in line_lower:
            # Simple author extraction
            if 'by ' in line_lower:
                author_part = line_lower.split('by ')[-1].strip()
                if len(author_part) < 50:  # Reasonable author length
                    author = author_part.title()
            break
    
    print(f"📚 Title: {title}")
    print(f"✍️  Author: {author}")
    print(f"📊 Content: {len(book_content):,} characters")
    print()
    
    # Check if this specific book was already processed
    selected_book_id = output_checker.generate_book_id(book_content, title)
    already_processed = output_checker.check_smart_processor_output_exists(selected_book_id)
    
    # Confirm processing
    print("⚡ This will:")
    if already_processed:
        print("  • Skip processing (already completed) and show existing results")
        print(f"  • Book ID: {selected_book_id}")
    else:
        print("  • Intelligently identify real characters (not random words)")
        print("  • Generate max 6 character portraits")
        print("  • Create max 4 key scene images") 
        print("  • Save JSON profiles for each")
        print("  • Take 2-5 minutes depending on book size")
    print()
    
    proceed = input("🚀 Start processing? (y/n): ").lower().strip()
    
    if proceed != 'y':
        print("👋 Processing cancelled")
        return
    
    # Initialize processor
    print("\n🔄 Starting smart processing...")
    processor = SmartBookProcessor()
    
    # Process the book
    result = processor.process_book(book_content, title, author)
    
    # Show results
    print("\n" + "=" * 50)
    
    if result["success"]:
        if result.get("skipped_processing"):
            print("⏩ USED EXISTING RESULTS!")
        else:
            print("✅ SMART PROCESSING COMPLETE!")
        print("=" * 50)
        
        book_info = result["book_info"]
        print(f"📚 Book: {book_info['title']}")
        print(f"✍️  Author: {book_info['author']}")
        print(f"📝 Words: {book_info['word_count']:,}")
        print()
        
        print("🎭 RESULTS:")
        print(f"👥 Real characters found: {result['characters_processed']}")
        print(f"🎨 Character portraits: {result['portraits_generated']}")
        print(f"🎬 Key scenes processed: {result['scenes_processed']}")
        print(f"🖼️  Scene images generated: {result['scene_images_generated']}")
        print(f"📁 Total files created: {result['total_files']}")
        print()
        
        print("🎯 CHARACTERS IDENTIFIED:")
        for name in result['character_names']:
            print(f"  • {name}")
        print()
        
        # Show file locations
        print("📂 GENERATED FILES:")
        generated_dir = Path("modelsbooknlp/generated")
        
        portraits_dir = generated_dir / "character_portraits"
        if portraits_dir.exists():
            portrait_files = list(portraits_dir.glob("*.jpg"))
            print(f"  🎨 Character Portraits: {len(portrait_files)} files")
            print(f"     📁 {portraits_dir}")
        
        scenes_dir = generated_dir / "scene_images" 
        if scenes_dir.exists():
            scene_files = list(scenes_dir.glob("*.jpg"))
            print(f"  🎬 Scene Images: {len(scene_files)} files")
            print(f"     📁 {scenes_dir}")
        
        profiles_dir = generated_dir / "profiles"
        if profiles_dir.exists():
            json_files = list(profiles_dir.glob("*.json"))
            print(f"  📋 JSON Profiles: {len(json_files)} files")
            print(f"     📁 {profiles_dir}")
        
        print()
        print("🎉 All files saved successfully!")
        print("🔍 Check the folders above to see your generated content!")
        
    else:
        print("❌ PROCESSING FAILED")
        print("=" * 50)
        print(f"Error: {result.get('error', 'Unknown error')}")
        print()
        print("💡 Common issues:")
        print("  • Book might not have clear character names")
        print("  • Text format might be unusual")
        print("  • Network issues with image generation")
        print("  • File encoding problems")
    
    print("\n👋 Processing complete!")


if __name__ == "__main__":
    main()