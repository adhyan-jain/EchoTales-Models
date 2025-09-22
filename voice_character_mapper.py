import json
import pandas as pd
import numpy as np
import random
import os
import shutil
from pathlib import Path
from typing import Dict, List

def calculate_ocean_distance(char_traits, voice_traits):
    """
    Calculate Euclidean distance between character and voice OCEAN traits.
    Lower distance means better match.
    """
    distance = 0
    ocean_keys = ['O', 'C', 'E', 'A', 'N']  # Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
    
    for key in ocean_keys:
        char_value = char_traits.get(key.lower(), 0.5)  # Default to 0.5 if missing
        voice_value = voice_traits.get(key, 0.5)  # Default to 0.5 if missing
        distance += (char_value - voice_value) ** 2
    
    return np.sqrt(distance)

def normalize_age_group(age_group):
    """
    Normalize age group names to match between character and voice data.
    """
    age_mapping = {
        'Young Adult': 'Young Adult',
        'Adult': 'Middle Aged',  # Map Adult to Middle Aged as closest match
        'Middle Aged': 'Middle Aged',
        'Older Adult': 'Older Adult',
        'Child': 'Young Adult',  # Map Child to Young Adult as closest available
    }
    return age_mapping.get(age_group, 'Middle Aged')

def normalize_gender(gender):
    """
    Normalize gender names to match between character and voice data.
    """
    gender_mapping = {
        'Male': 'Male',
        'Female': 'Female',
        'Unknown': None,  # Will be handled specially
        'M': 'Male',
        'F': 'Female'
    }
    return gender_mapping.get(gender, None)

def find_best_voice_match(character, voices_df):
    """
    Find the best voice match for a character based on gender, age, and OCEAN traits.
    """
    char_gender = normalize_gender(character['gender'])
    char_age = normalize_age_group(character['age_group'])
    
    # If gender is unknown, we'll try both male and female voices
    if char_gender is None:
        # Randomly pick a gender for unknown characters
        char_gender = random.choice(['Male', 'Female'])
    
    # Filter voices by gender and age
    filtered_voices = voices_df[
        (voices_df['gender'] == char_gender) & 
        (voices_df['age_group'] == char_age)
    ].copy()
    
    # If no exact age match, try broader age matching
    if filtered_voices.empty:
        # Just match by gender
        filtered_voices = voices_df[voices_df['gender'] == char_gender].copy()
    
    # If still no match (shouldn't happen with our data), use all voices
    if filtered_voices.empty:
        filtered_voices = voices_df.copy()
    
    # Calculate OCEAN distance for each filtered voice
    char_ocean = {
        'O': character['personality']['openness'],
        'C': character['personality']['conscientiousness'], 
        'E': character['personality']['extraversion'],
        'A': character['personality']['agreeableness'],
        'N': character['personality']['neuroticism']
    }
    
    distances = []
    for idx, voice_row in filtered_voices.iterrows():
        voice_ocean = {
            'O': voice_row['O'],
            'C': voice_row['C'],
            'E': voice_row['E'], 
            'A': voice_row['A'],
            'N': voice_row['N']
        }
        distance = calculate_ocean_distance(char_ocean, voice_ocean)
        distances.append(distance)
    
    filtered_voices['ocean_distance'] = distances
    
    # Find the best match (lowest distance)
    best_match = filtered_voices.loc[filtered_voices['ocean_distance'].idxmin()]
    
    return best_match

def generate_character_looks(character_name: str, gender: str, age_group: str, personality: Dict[str, float]) -> Dict[str, str]:
    """
    Generate comprehensive character appearance descriptions based on character traits.
    """
    # Height variations by gender and age
    height_ranges = {
        'male': {
            'child': 'Short (4\'0" - 4\'6")',
            'young_adult': 'Average to tall (5\'8" - 6\'2")',
            'adult': 'Average to tall (5\'8" - 6\'2")',
            'elderly': 'Medium to tall (5\'6" - 6\'0")'
        },
        'female': {
            'child': 'Short (3\'8" - 4\'4")',
            'young_adult': 'Average (5\'4" - 5\'8")',
            'adult': 'Average (5\'4" - 5\'8")',
            'elderly': 'Medium (5\'2" - 5\'6")'
        },
        'unknown': {
            'child': 'Short (4\'0" - 4\'6")',
            'young_adult': 'Average (5\'6" - 5\'10")',
            'adult': 'Average (5\'6" - 5\'10")',
            'elderly': 'Medium (5\'4" - 5\'8")'
        }
    }
    
    # Build variations by personality traits
    build_options = {
        'high_extraversion': ['Athletic', 'Strong', 'Well-built', 'Robust'],
        'low_extraversion': ['Slender', 'Lean', 'Delicate', 'Slight'],
        'high_conscientiousness': ['Well-maintained', 'Fit', 'Toned', 'Disciplined build'],
        'high_neuroticism': ['Tense', 'Wiry', 'Nervous energy', 'Restless build'],
        'default': ['Average build', 'Medium build', 'Balanced physique', 'Proportioned']
    }
    
    # Hair colors and styles
    hair_colors = {
        'common': ['Dark brown', 'Light brown', 'Black', 'Blonde', 'Auburn'],
        'uncommon': ['Red', 'Copper', 'Chestnut', 'Ash brown', 'Golden blonde'],
        'aging': ['Gray', 'Silver', 'Gray with streaks', 'Salt and pepper', 'White']
    }
    
    hair_styles = {
        'male': {
            'young': ['Messy', 'Tousled', 'Short and neat', 'Slightly wavy', 'Styled'],
            'adult': ['Well-groomed', 'Professional cut', 'Neat', 'Classic style', 'Practical'],
            'elderly': ['Thinning', 'Receding', 'Neatly combed', 'Sparse', 'Conservative']
        },
        'female': {
            'young': ['Long and flowing', 'Wavy', 'Straight and silky', 'Curly', 'Stylishly cut'],
            'adult': ['Shoulder-length', 'Elegant updo', 'Professional style', 'Layered', 'Sophisticated'],
            'elderly': ['Short and practical', 'Neatly styled', 'Soft curls', 'Conservative cut', 'Gray and dignified']
        },
        'unknown': {
            'young': ['Neat', 'Practical', 'Styled', 'Well-kept', 'Casual'],
            'adult': ['Professional', 'Well-maintained', 'Classic', 'Practical', 'Tidy'],
            'elderly': ['Conservative', 'Neat', 'Traditional', 'Well-groomed', 'Dignified']
        }
    }
    
    # Eye colors
    eye_colors = ['Brown', 'Blue', 'Green', 'Hazel', 'Gray', 'Dark brown', 'Light blue', 'Emerald green', 'Amber']
    
    # Skin tones
    skin_tones = ['Fair', 'Light', 'Medium', 'Olive', 'Tan', 'Dark', 'Pale', 'Rosy', 'Weathered']
    
    # Clothing styles based on personality
    clothing_styles = {
        'high_openness': ['Artistic clothing', 'Creative fashion', 'Unique style', 'Bohemian attire', 'Expressive dress'],
        'high_conscientiousness': ['Professional attire', 'Well-tailored clothes', 'Neat and organized dress', 'Business formal', 'Immaculate presentation'],
        'high_extraversion': ['Bold fashion choices', 'Eye-catching attire', 'Confident style', 'Fashionable dress', 'Statement pieces'],
        'high_agreeableness': ['Comfortable clothing', 'Approachable style', 'Soft colors', 'Friendly appearance', 'Warm fashion'],
        'default': ['Practical clothing', 'Simple attire', 'Casual dress', 'Comfortable style', 'Understated fashion']
    }
    
    # Determine character traits
    gender_key = gender.lower() if gender.lower() in ['male', 'female'] else 'unknown'
    # Convert voice-compatible age groups to internal keys
    age_mapping = {
        'young adult': 'young_adult',
        'middle aged': 'adult', 
        'older adult': 'elderly'
    }
    age_key = age_mapping.get(age_group.lower(), 'adult')
    
    # Generate height
    height = height_ranges[gender_key].get(age_key, 'Average')
    
    # Generate build based on personality
    build = 'Average build'
    if personality['extraversion'] > 0.7:
        build = random.choice(build_options['high_extraversion'])
    elif personality['extraversion'] < 0.4:
        build = random.choice(build_options['low_extraversion'])
    elif personality['conscientiousness'] > 0.7:
        build = random.choice(build_options['high_conscientiousness'])
    elif personality['neuroticism'] > 0.7:
        build = random.choice(build_options['high_neuroticism'])
    else:
        build = random.choice(build_options['default'])
    
    # Generate hair
    if age_key == 'elderly':
        hair_color = random.choice(hair_colors['aging'])
    elif personality['openness'] > 0.7:
        hair_color = random.choice(hair_colors['uncommon'])
    else:
        hair_color = random.choice(hair_colors['common'])
    
    # Hair style based on gender and age
    age_style_key = 'young' if age_key in ['child', 'young_adult'] else 'adult' if age_key == 'adult' else 'elderly'
    hair_style = random.choice(hair_styles[gender_key][age_style_key])
    
    # Generate other features
    eye_color = random.choice(eye_colors)
    skin_tone = random.choice(skin_tones)
    
    # Generate clothing style based on personality
    clothing = 'Practical clothing'
    if personality['openness'] > 0.7:
        clothing = random.choice(clothing_styles['high_openness'])
    elif personality['conscientiousness'] > 0.7:
        clothing = random.choice(clothing_styles['high_conscientiousness'])
    elif personality['extraversion'] > 0.7:
        clothing = random.choice(clothing_styles['high_extraversion'])
    elif personality['agreeableness'] > 0.7:
        clothing = random.choice(clothing_styles['high_agreeableness'])
    else:
        clothing = random.choice(clothing_styles['default'])
    
    # Generate distinctive features based on personality and context
    distinctive_features = []
    if personality['openness'] > 0.7:
        distinctive_features.extend(['Creative aura', 'Expressive eyes', 'Artistic presence'])
    if personality['conscientiousness'] > 0.7:
        distinctive_features.extend(['Well-groomed appearance', 'Neat presentation', 'Organized bearing'])
    if personality['extraversion'] > 0.7:
        distinctive_features.extend(['Confident posture', 'Engaging smile', 'Charismatic presence'])
    if personality['agreeableness'] > 0.7:
        distinctive_features.extend(['Kind eyes', 'Warm expression', 'Approachable demeanor'])
    if personality['neuroticism'] > 0.7:
        distinctive_features.extend(['Intense gaze', 'Expressive features', 'Dynamic energy'])
    
    # Add some randomness and character-specific features
    general_features = ['Intelligent eyes', 'Strong jawline', 'Gentle smile', 'Piercing gaze', 'Graceful movements', 'Confident stance', 'Mysterious aura', 'Noble bearing']
    distinctive_features.extend(random.sample(general_features, min(2, len(general_features))))
    
    # Create the comprehensive physical description
    age_descriptor = {
        'child': 'young',
        'young_adult': 'young adult',
        'adult': 'adult', 
        'elderly': 'older individual'
    }.get(age_key, 'person')
    
    gender_descriptor = {
        'male': 'man',
        'female': 'woman',
        'unknown': 'individual'
    }.get(gender_key, 'person')
    
    physical_description = f"A {build.lower()} {age_descriptor} with {hair_color.lower()} hair that is typically {hair_style.lower()}. They have {eye_color.lower()} eyes and {skin_tone.lower()} skin. {character_name} is known for their {', '.join(distinctive_features[:3]).lower()} and typically dresses in {clothing.lower()}. Their overall presence suggests {random.choice(['confidence', 'mystery', 'warmth', 'intelligence', 'strength', 'grace'])} and they carry themselves with {random.choice(['poise', 'determination', 'quiet dignity', 'natural charm', 'understated elegance'])}."
    
    return {
        'physical_description': physical_description,
        'height': height,
        'build': build,
        'hair_color': hair_color,
        'hair_style': hair_style,
        'eye_color': eye_color,
        'skin_tone': skin_tone,
        'distinctive_features': ', '.join(distinctive_features[:4]),
        'typical_clothing': clothing
    }

def create_character_voice_mapping(voices_source_dir=None):
    """
    Main function to create character-to-voice mapping and copy audio files.
    
    Args:
        voices_source_dir (str, optional): Directory containing voice files. 
                                          If None, will skip file copying.
    """
    # Load character data
    print("Loading character data...")
    with open('characters/advanced_characters.json', 'r') as f:
        character_data = json.load(f)
    
    # Load voice data
    print("Loading voice data...")
    voices_df = pd.read_csv('final.csv')
    
    # Create audio directory if it doesn't exist
    audio_dir = Path('audio')
    audio_dir.mkdir(exist_ok=True)
    
    # Create mapping results
    mapping_results = []
    files_copied = 0
    files_not_found = 0
    
    print(f"Processing {len(character_data['characters'])} characters...")
    
    for character in character_data['characters']:
        print(f"Processing character: {character['name']}")
        
        # Find best voice match
        best_voice = find_best_voice_match(character, voices_df)
        
        # Generate character looks
        character_looks = generate_character_looks(
            character['name'],
            character['gender'],
            character['age_group'],
            character['personality']
        )
        
        # Create character folder
        # Clean filename by removing/replacing invalid characters
        char_name = character['name']
        # Replace problematic characters with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            char_name = char_name.replace(char, '_')
        # Also replace common problematic patterns
        char_name = char_name.replace('*', 'star').replace('â€™', "'")
        # Remove leading/trailing dots and spaces
        char_name = char_name.strip('. ')
        # Limit length to avoid path issues
        if len(char_name) > 100:
            char_name = char_name[:100]
        # Ensure it's not empty
        if not char_name or char_name == '.':
            char_name = f"character_{character['character_id']}"
        
        char_folder = audio_dir / char_name
        char_folder.mkdir(exist_ok=True)
        
        # Get voice filename
        voice_filename = best_voice['filename']
        
        # Copy voice file if source directory is provided
        file_copied = False
        if voices_source_dir:
            voice_source_path = Path(voices_source_dir) / voice_filename
            # Always save as 'voice.mp3' regardless of original format
            voice_dest_path = char_folder / "voice.mp3"
            
            if voice_source_path.exists():
                shutil.copy2(voice_source_path, voice_dest_path)
                print(f"  Copied {voice_filename} to {char_folder}/voice.mp3")
                files_copied += 1
                file_copied = True
            else:
                print(f"  Warning: Voice file {voice_filename} not found in {voices_source_dir}")
                files_not_found += 1
        
        # Store mapping information
        mapping_info = {
            'character_id': character['character_id'],
            'character_name': character['name'],
            'character_gender': character['gender'],
            'character_age': character['age_group'],
            'character_looks': character_looks,
            'character_ocean': character['personality'],
            'voice_filename': voice_filename,
            'voice_gender': best_voice['gender'],
            'voice_age': best_voice['age_group'],
            'voice_ocean': {
                'openness': best_voice['O'],
                'conscientiousness': best_voice['C'],
                'extraversion': best_voice['E'],
                'agreeableness': best_voice['A'],
                'neuroticism': best_voice['N']
            },
            'ocean_distance': best_voice['ocean_distance'],
            'character_folder': str(char_folder),
            'file_copied': file_copied
        }
        
        mapping_results.append(mapping_info)
    
    # Save mapping results
    print("Saving mapping results...")
    with open('character_voice_mapping.json', 'w') as f:
        json.dump(mapping_results, f, indent=2)
    
    # Create summary statistics
    print("\nMapping Summary:")
    print(f"Total characters mapped: {len(mapping_results)}")
    
    # Gender matching stats
    perfect_gender_matches = sum(1 for m in mapping_results 
                               if m['character_gender'] == m['voice_gender'])
    print(f"Perfect gender matches: {perfect_gender_matches}/{len(mapping_results)}")
    
    # Age matching stats  
    perfect_age_matches = sum(1 for m in mapping_results 
                            if m['character_age'] == m['voice_age'])
    print(f"Perfect age matches: {perfect_age_matches}/{len(mapping_results)}")
    
    # Average OCEAN distance
    avg_distance = np.mean([m['ocean_distance'] for m in mapping_results])
    print(f"Average OCEAN distance: {avg_distance:.4f}")
    
    # File copying stats
    if voices_source_dir:
        print(f"Files successfully copied: {files_copied}")
        print(f"Files not found: {files_not_found}")
    else:
        print("No voice source directory provided - skipped file copying")
    
    print(f"\nMapping complete! Results saved to character_voice_mapping.json")
    print(f"Character folders created in the 'audio' directory")

def find_voice_directory():
    """
    Try to automatically find a directory containing voice files.
    """
    possible_dirs = [
        'voices',
        'audio_files', 
        'sound_files',
        'voice_files',
        '../voices',
        '../audio_files'
    ]
    
    for dir_path in possible_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            # Check if it contains audio files
            audio_files = list(path.glob('*.mp3')) + list(path.glob('*.wav')) + list(path.glob('*.m4a'))
            if audio_files:
                print(f"Found voice directory: {dir_path}")
                return str(path)
    
    return None

if __name__ == "__main__":
    # Try to automatically find voice directory
    voice_dir = find_voice_directory()
    
    if voice_dir:
        print(f"Using voice directory: {voice_dir}")
        create_character_voice_mapping(voice_dir)
    else:
        print("No voice directory found. Creating mapping without copying files.")
        print("To copy voice files, please specify the directory containing voice files:")
        print("create_character_voice_mapping('path/to/your/voice/files')")
        create_character_voice_mapping()
