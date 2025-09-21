#!/usr/bin/env python3
"""
Advanced Character Processing System - Example Usage
Demonstrates the sophisticated character processing capabilities
"""

import json
from pathlib import Path
from advanced_character_processor import AdvancedCharacterProcessor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_advanced_processing():
    """Demonstrate the advanced character processing system"""
    print("="*80)
    print("ADVANCED CHARACTER PROCESSING SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize the advanced processor
    processor = AdvancedCharacterProcessor(
        output_dir="modelsbooknlp/output",
        characters_dir="characters"
    )
    
    book_id = "samples"
    
    try:
        print(f"\nProcessing book: '{book_id}'")
        print("="*50)
        
        # Process the book with advanced features
        output_path, characters = processor.process_book(book_id)
        
        print(f"\nAdvanced processing complete!")
        print(f"Characters saved to: {output_path}")
        print(f"Total characters processed: {len(characters)}")
        
        # Display detailed results
        print(f"\nDETAILED CHARACTER ANALYSIS")
        print("="*50)
        
        for i, char in enumerate(characters[:10], 1):  # Show top 10 characters
            print(f"\n{i}. {char['name']} (ID: {char['character_id']})")
            print(f"   Statistics:")
            print(f"      • Mentions: {char['mention_count']}")
            print(f"      • Importance Score: {char['character_analysis']['importance_score']}")
            print(f"      • Dialogue Lines: {char['character_analysis']['total_dialogue_lines']}")
            print(f"      • Narrator Mentions: {char['character_analysis']['total_narrator_mentions']}")
            
            print(f"   Personality Analysis (OCEAN):")
            personality = char['personality']
            print(f"      • Openness: {personality['O']:.2f}")
            print(f"      • Conscientiousness: {personality['C']:.2f}")
            print(f"      • Extraversion: {personality['E']:.2f}")
            print(f"      • Agreeableness: {personality['A']:.2f}")
            print(f"      • Neuroticism: {personality['N']:.2f}")
            print(f"      • Confidence: {char['character_analysis']['personality_analysis_confidence']}")
            
            print(f"   Demographics:")
            print(f"      • Gender: {char['gender']} (method: {char['character_analysis']['gender_classification_method']})")
            print(f"      • Age Group: {char['age_group']}")
            
            # Show character consolidation info
            if len(char['character_analysis']['merged_from']) > 1:
                print(f"   Character Consolidation:")
                print(f"      • Merged from: {', '.join(char['character_analysis']['merged_from'])}")
            
            # Show sample quotes
            if char['quotes']:
                print(f"   Sample Dialogue:")
                for j, quote in enumerate(char['quotes'][:2], 1):
                    print(f"      {j}. \"{quote['quote_text'][:80]}{'...' if len(quote['quote_text']) > 80 else ''}\"")
                    print(f"         Confidence: {quote['speaker_confidence']:.2f}, Type: {quote['context_type']}")
            
            # Show narrator mentions
            if char['narrator_mentions']:
                print(f"   Sample Narrator Mentions:")
                for j, mention in enumerate(char['narrator_mentions'][:2], 1):
                    print(f"      {j}. {mention['mention_text'][:80]}{'...' if len(mention['mention_text']) > 80 else ''}")
                    print(f"         Type: {mention['mention_type']}")
        
        # Show processing summary
        print(f"\nPROCESSING SUMMARY")
        print("="*50)
        
        total_dialogue = sum(char['character_analysis']['total_dialogue_lines'] for char in characters)
        total_narrator_mentions = sum(char['character_analysis']['total_narrator_mentions'] for char in characters)
        total_merged = sum(1 for char in characters if len(char['character_analysis']['merged_from']) > 1)
        
        print(f"• Total Characters: {len(characters)}")
        print(f"• Total Dialogue Lines: {total_dialogue}")
        print(f"• Total Narrator Mentions: {total_narrator_mentions}")
        print(f"• Characters with Merged Identities: {total_merged}")
        print(f"• Average Personality Confidence: {sum(char['character_analysis']['personality_analysis_confidence'] for char in characters) / len(characters):.2f}")
        
        # Gender distribution
        gender_dist = {}
        for char in characters:
            gender = char['gender']
            gender_dist[gender] = gender_dist.get(gender, 0) + 1
        
        print(f"• Gender Distribution: {gender_dist}")
        
        # Age group distribution
        age_dist = {}
        for char in characters:
            age = char['age_group']
            age_dist[age] = age_dist.get(age, 0) + 1
        
        print(f"• Age Group Distribution: {age_dist}")
        
        return output_path, characters
        
    except Exception as e:
        logger.error(f"Advanced processing failed: {e}")
        raise

def demonstrate_character_consolidation():
    """Demonstrate character consolidation features"""
    print(f"\nCHARACTER CONSOLIDATION DEMONSTRATION")
    print("="*50)
    
    processor = AdvancedCharacterProcessor()
    
    # Example of character consolidation
    test_characters = {
        1: {'name': 'Klein', 'mention_count': 50, 'context_texts': [], 'all_mentions': []},
        2: {'name': 'Klein Moretti', 'mention_count': 30, 'context_texts': [], 'all_mentions': []},
        3: {'name': 'he', 'mention_count': 100, 'context_texts': [], 'all_mentions': []},
        4: {'name': 'Zhou Mingrui', 'mention_count': 20, 'context_texts': [], 'all_mentions': []}
    }
    
    print("Before consolidation:")
    for char_id, char_data in test_characters.items():
        print(f"  {char_id}: {char_data['name']} ({char_data['mention_count']} mentions)")
    
    consolidated = processor.consolidate_characters(test_characters)
    
    print("\nAfter consolidation:")
    for char_id, char_data in consolidated.items():
        print(f"  {char_id}: {char_data['name']} ({char_data['mention_count']} mentions)")
        if 'merged_from' in char_data and len(char_data['merged_from']) > 1:
            print(f"    Merged from: {', '.join(char_data['merged_from'])}")

def demonstrate_quote_attribution():
    """Demonstrate quote attribution analysis"""
    print(f"\nQUOTE ATTRIBUTION DEMONSTRATION")
    print("="*50)
    
    processor = AdvancedCharacterProcessor()
    
    # Example quote analysis
    test_quotes = [
        {
            'quote_text': '"Hello, how are you?"',
            'mention_phrase': 'Klein',
            'context': 'Klein said, "Hello, how are you?" to his friend.'
        },
        {
            'quote_text': '"I think this is interesting"',
            'mention_phrase': 'Klein',
            'context': 'Klein thought to himself, "I think this is interesting"'
        },
        {
            'quote_text': '"What happened?"',
            'mention_phrase': 'he',
            'context': 'He asked, "What happened?" with concern.'
        }
    ]
    
    for i, quote in enumerate(test_quotes, 1):
        print(f"\n{i}. Quote: {quote['quote_text']}")
        print(f"   Mention: {quote['mention_phrase']}")
        print(f"   Context: {quote['context']}")
        
        # Simulate context analysis
        context_analysis = {
            'confidence': 0.8 if 'said' in quote['context'] else 0.3,
            'mention_type': 'dialogue_tag' if 'said' in quote['context'] else 'internal_thought',
            'has_quotes': True
        }
        
        is_dialogue = processor.is_actual_dialogue(
            quote['quote_text'], 
            quote['mention_phrase'], 
            context_analysis
        )
        
        print(f"   Analysis: {'Dialogue' if is_dialogue else 'Narrator Description'}")
        print(f"   Confidence: {context_analysis['confidence']:.2f}")
        print(f"   Type: {context_analysis['mention_type']}")

def main():
    """Main demonstration function"""
    try:
        # Run the main demonstration
        output_path, characters = demonstrate_advanced_processing()
        
        # Run additional demonstrations
        demonstrate_character_consolidation()
        demonstrate_quote_attribution()
        
        print(f"\nDEMONSTRATION COMPLETE!")
        print("="*50)
        print(f"Results saved to: {output_path}")
        print(f"Processed {len(characters)} characters with advanced analysis")
        print(f"Features demonstrated:")
        print(f"   • Pronoun character filtering")
        print(f"   • Character consolidation and merging")
        print(f"   • Advanced quote attribution analysis")
        print(f"   • Narrator mention tracking")
        print(f"   • Multi-method gender classification")
        print(f"   • Pre-trained personality analysis")
        print(f"   • Context-aware character analysis")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()
