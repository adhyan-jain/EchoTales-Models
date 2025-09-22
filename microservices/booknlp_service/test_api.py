#!/usr/bin/env python3
"""
Test script for BookNLP API service
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8002"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint failed: {e}")
        return False

def test_book_processing():
    """Test book processing with sample text"""
    print("\nTesting book processing...")
    
    # Sample text for testing
    sample_text = """
    Chapter 1: The Beginning
    
    John was a young man living in the city. He worked as a software engineer and had a passion for books.
    
    "I love reading," John said to his friend Mary. "Books transport you to different worlds."
    
    Mary smiled and replied, "I agree completely. Literature has the power to change lives."
    
    They were sitting in a small cafe downtown, discussing their favorite novels. The afternoon sun streamed through the windows, creating a warm atmosphere.
    
    John picked up his coffee cup and took a sip. "Have you read the latest mystery novel by that famous author?" he asked.
    
    "Which one?" Mary inquired, her eyes lighting up with interest.
    
    "The one about the detective who solves crimes using literary clues," John explained enthusiastically.
    
    Chapter 2: The Discovery
    
    The next day, John discovered an old bookstore hidden in an alley. The owner, Mr. Thompson, was an elderly gentleman with kind eyes.
    
    "Welcome to my humble store," Mr. Thompson greeted John warmly. "Are you looking for anything particular?"
    
    John browsed the shelves filled with rare and antique books. "I'm interested in mystery novels," he said.
    
    Mr. Thompson nodded knowingly. "Ah, a fellow mystery enthusiast. I have just the thing for you."
    
    He led John to a special section of the store where the most intriguing books were kept.
    """
    
    request_data = {
        "text": sample_text,
        "book_id": "test_book_001",
        "title": "Test Novel",
        "author": "Test Author"
    }
    
    try:
        print("Sending processing request...")
        response = requests.post(f"{BASE_URL}/process", json=request_data, timeout=300)  # 5 minute timeout
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Processing successful!")
            print(f"Book ID: {result['book_id']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            print(f"Output directory: {result['output_directory']}")
            print(f"Files generated: {result['files_generated']}")
            print(f"Entities count: {result.get('entities_count', 'N/A')}")
            print(f"Quotes count: {result.get('quotes_count', 'N/A')}")
            print(f"Characters count: {result.get('characters_count', 'N/A')}")
            
            return True, result['book_id']
        else:
            print(f"Processing failed: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print("Processing request timed out (this is normal for large texts)")
        return False, None
    except Exception as e:
        print(f"Processing request failed: {e}")
        return False, None

def test_get_results(book_id):
    """Test getting processing results"""
    print(f"\nTesting results retrieval for book_id: {book_id}...")
    try:
        response = requests.get(f"{BASE_URL}/results/{book_id}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Results retrieved successfully!")
            print(f"Book ID: {result['book_id']}")
            print(f"Files available: {result['files_generated']}")
            print(f"Entities count: {result.get('entities_count', 'N/A')}")
            return True
        else:
            print(f"Failed to get results: {response.text}")
            return False
            
    except Exception as e:
        print(f"Results request failed: {e}")
        return False

def test_list_books():
    """Test listing all processed books"""
    print("\nTesting book listing...")
    try:
        response = requests.get(f"{BASE_URL}/books")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Books listed successfully!")
            print(f"Total books: {result['total_books']}")
            for book in result['books']:
                print(f"  - {book['book_id']}: {len(book['files_generated'])} files")
            return True
        else:
            print(f"Failed to list books: {response.text}")
            return False
            
    except Exception as e:
        print(f"List books request failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("BookNLP API Test Suite")
    print("=" * 50)
    
    # Check if API is running
    print("Checking if API is running...")
    if not test_health_check():
        print("❌ API is not running. Please start the service first:")
        print("   cd microservices/booknlp_service")
        print("   python app.py")
        return
    
    print("✅ API is running!")
    
    # Test all endpoints
    tests = [
        ("Root endpoint", test_root_endpoint),
        ("Health check", test_health_check),
    ]
    
    # Test basic endpoints
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        success = test_func()
        if success:
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    # Test book processing
    print(f"\n{'='*20}")
    success, book_id = test_book_processing()
    if success:
        print("✅ Book processing passed")
        
        # Test results retrieval if processing was successful
        if book_id:
            if test_get_results(book_id):
                print("✅ Results retrieval passed")
            else:
                print("❌ Results retrieval failed")
    else:
        print("❌ Book processing failed")
    
    # Test book listing
    print(f"\n{'='*20}")
    if test_list_books():
        print("✅ Book listing passed")
    else:
        print("❌ Book listing failed")
    
    print(f"\n{'='*50}")
    print("Test suite completed!")
    print("Note: Some tests may take several minutes due to BookNLP processing time.")

if __name__ == "__main__":
    main()