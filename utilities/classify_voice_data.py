
import os
import json
import librosa
import numpy as np
from pathlib import Path
import logging
import traceback
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_audio_file(file_path):
    '''Analyze audio file and extract features with robust error handling'''
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None
            
        # Load audio with error handling
        try:
            audio, sr = librosa.load(file_path, sr=22050)
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            return None
        
        # Check if audio is valid
        if len(audio) == 0:
            logger.error(f"Empty audio file: {file_path}")
            return None
            
        # Extract basic features
        duration = len(audio) / sr
        
        # Extract pitch with error handling
        try:
            pitch = librosa.yin(audio, fmin=50, fmax=500)
            # Remove NaN values and handle edge cases
            pitch = pitch[~np.isnan(pitch)]
            if len(pitch) > 0:
                pitch_mean = float(np.mean(pitch))
                pitch_std = float(np.std(pitch))
            else:
                pitch_mean = 0.0
                pitch_std = 0.0
        except Exception as e:
            logger.warning(f"Failed to extract pitch from {file_path}: {e}")
            pitch_mean = 0.0
            pitch_std = 0.0
        
        # Extract energy with error handling
        try:
            rms = librosa.feature.rms(y=audio)
            energy_mean = float(np.mean(rms))
        except Exception as e:
            logger.warning(f"Failed to extract energy from {file_path}: {e}")
            energy_mean = 0.0
        
        # Extract spectral centroid with error handling
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_centroid_mean = float(np.mean(spectral_centroid))
        except Exception as e:
            logger.warning(f"Failed to extract spectral centroid from {file_path}: {e}")
            spectral_centroid_mean = 0.0
        
        # Extract MFCCs with error handling
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = mfccs.mean(axis=1).tolist()
        except Exception as e:
            logger.warning(f"Failed to extract MFCCs from {file_path}: {e}")
            mfcc_mean = [0.0] * 13
        
        features = {
            'duration': duration,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'energy_mean': energy_mean,
            'spectral_centroid': spectral_centroid_mean,
            'mfcc_mean': mfcc_mean,
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Critical error analyzing {file_path}: {e}")
        logger.error(traceback.format_exc())
        return None

def classify_voice_dataset():
    '''Classify all voice files in the dataset with progress tracking'''
    voice_path = Path('data/voice_dataset')
    results = {
        'male_actors': {},
        'female_actors': {},
        'metadata': {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'errors': []
        }
    }
    
    # First pass: count total files
    print("Counting audio files...")
    all_files = []
    
    # Count male actors
    male_path = voice_path / 'male_actors'
    if male_path.exists():
        for actor_dir in male_path.iterdir():
            if actor_dir.is_dir():
                for audio_file in actor_dir.glob('*.wav'):
                    all_files.append(('male', actor_dir.name, audio_file))
    
    # Count female actors
    female_path = voice_path / 'female_actors'
    if female_path.exists():
        for actor_dir in female_path.iterdir():
            if actor_dir.is_dir():
                for audio_file in actor_dir.glob('*.wav'):
                    all_files.append(('female', actor_dir.name, audio_file))
    
    results['metadata']['total_files'] = len(all_files)
    print(f"Found {len(all_files)} audio files to process")
    
    # Initialize actor entries
    for gender, actor_name, _ in all_files:
        if gender == 'male':
            if actor_name not in results['male_actors']:
                results['male_actors'][actor_name] = {
                    'gender': 'male',
                    'files': {},
                    'characteristics': {}
                }
        else:
            if actor_name not in results['female_actors']:
                results['female_actors'][actor_name] = {
                    'gender': 'female',
                    'files': {},
                    'characteristics': {}
                }
    
    # Process files with progress bar
    print("\nProcessing audio files...")
    with tqdm(total=len(all_files), desc="Analyzing audio", unit="file") as pbar:
        for gender, actor_name, audio_file in all_files:
            pbar.set_description(f"Processing {actor_name}/{audio_file.name}")
            
            features = analyze_audio_file(audio_file)
            
            if features:
                if gender == 'male':
                    results['male_actors'][actor_name]['files'][audio_file.name] = features
                else:
                    results['female_actors'][actor_name]['files'][audio_file.name] = features
                results['metadata']['processed_files'] += 1
            else:
                results['metadata']['failed_files'] += 1
                error_info = {
                    'file': str(audio_file),
                    'actor': actor_name,
                    'gender': gender
                }
                results['metadata']['errors'].append(error_info)
            
            pbar.update(1)
    
    # Save results
    output_file = voice_path / 'voice_classification.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nVoice classification complete!")
    print(f"Total files: {results['metadata']['total_files']}")
    print(f"Processed successfully: {results['metadata']['processed_files']}")
    print(f"Failed: {results['metadata']['failed_files']}")
    
    if results['metadata']['failed_files'] > 0:
        print(f"\nFailed files:")
        for error in results['metadata']['errors'][:10]:  # Show first 10 errors
            print(f"  - {error['file']} ({error['actor']}, {error['gender']})")
        if len(results['metadata']['errors']) > 10:
            print(f"  ... and {len(results['metadata']['errors']) - 10} more")
    
    success_rate = (results['metadata']['processed_files'] / results['metadata']['total_files']) * 100
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    classify_voice_dataset()
