#!/usr/bin/env python3
"""
Fixed Chapter Audio Creator
Converts MP3-disguised-as-WAV files and concatenates them properly
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import tempfile

def convert_mp3_to_wav(input_file, output_file):
    """Convert MP3 file to WAV using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite existing files
            '-i', str(input_file),
            '-acodec', 'pcm_s16le',  # Standard WAV format
            '-ar', '22050',  # Sample rate
            '-ac', '1',  # Mono
            str(output_file)
        ]
        
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        
        if result.returncode == 0:
            return True
        else:
            print(f"  ❌ ffmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        return False

def concatenate_wav_files(wav_files, output_file):
    """Concatenate WAV files using ffmpeg"""
    try:
        # Create a temporary file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for wav_file in wav_files:
                f.write(f"file '{wav_file.absolute()}'\n")
            filelist_path = f.name
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',
            str(output_file)
        ]
        
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        
        # Clean up temp file
        os.unlink(filelist_path)
        
        if result.returncode == 0:
            return True
        else:
            print(f"❌ Concatenation error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Concatenation failed: {e}")
        return False

def create_voice_summary():
    """Create a summary of voice assignments"""
    audio_dir = Path("generated_audio")
    line_files = list(audio_dir.glob("ch0001_line*.wav"))
    
    if not line_files:
        print("❌ No line files to analyze")
        return
    
    # Analyze voice usage from filenames
    voice_usage = defaultdict(list)
    character_lines = defaultdict(int)
    
    for file_path in line_files:
        filename = file_path.name
        # Extract info from filename: ch0001_line###_line_######_char_###.wav
        parts = filename.replace('.wav', '').split('_')
        if len(parts) >= 4:
            line_num = parts[1].replace('line', '')
            char_id = parts[-1]
            
            # Map character IDs to names and voices
            if char_id == 'narrator':
                char_name = 'Narrator'
                voice = 'en-US-ChristopherNeural'
            elif char_id == '000':
                char_name = 'Character (I)'
                voice = 'en-US-RogerNeural'
            elif char_id == '415':
                char_name = 'Zhou Mingrui'
                voice = 'en-US-ChristopherNeural'  # The cool voice you love!
            elif char_id == '416':
                char_name = 'Pa'
                voice = 'en-US-RogerNeural'
            elif char_id == '417':
                char_name = 'Klein'
                voice = 'en-US-ChristopherNeural'
            else:
                char_name = f'Character_{char_id}'
                voice = 'en-US-SteffanNeural'
            
            voice_usage[char_name].append(int(line_num))
            character_lines[char_name] += 1
    
    # Create voice summary
    print(f"\n🗣️ VOICE ASSIGNMENTS SUMMARY")
    print("=" * 40)
    
    total_lines = sum(character_lines.values())
    
    for char_name, line_count in sorted(character_lines.items(), key=lambda x: x[1], reverse=True):
        percentage = (line_count / total_lines) * 100
        lines = sorted(voice_usage[char_name])
        line_ranges = f"Lines {lines[0]}-{lines[-1]}" if lines else "No lines"
        
        # Highlight Zhou Mingrui since you love that voice
        emoji = "🌟" if char_name == "Zhou Mingrui" else "🎤"
        print(f"{emoji} {char_name:<20} | {line_count:3d} lines ({percentage:4.1f}%) | {line_ranges}")
    
    print(f"\n📊 CHAPTER STATISTICS")
    print("-" * 25)
    print(f"Total processed lines: {total_lines}")
    print(f"Main characters: {len([c for c, count in character_lines.items() if count > 5])}")
    print(f"Speaking characters: {len(character_lines)}")
    
    # Save summary to file
    summary_data = {
        'chapter': 1,
        'total_lines': total_lines,
        'character_line_counts': dict(character_lines),
        'voice_assignments': {
            char: {
                'lines': sorted(voice_usage[char]),
                'count': character_lines[char],
                'percentage': round((character_lines[char] / total_lines) * 100, 1),
                'voice': 'en-US-ChristopherNeural' if char == 'Zhou Mingrui' else 'various'
            }
            for char in character_lines.keys()
        }
    }
    
    summary_file = audio_dir / "chapter_0001_voice_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Voice summary saved to: {summary_file.name}")

def main():
    """Main function"""
    print("🎭 EchoTales Fixed Chapter Audio Creator")
    print("=" * 50)
    
    audio_dir = Path("generated_audio")
    if not audio_dir.exists():
        print("❌ No generated_audio directory found")
        return
    
    # Find all line audio files
    line_files = sorted(list(audio_dir.glob("ch0001_line*.wav")))
    
    if not line_files:
        print("❌ No line audio files found")
        return
    
    print(f"📁 Found {len(line_files)} line audio files")
    
    # Create voice summary first
    print("📋 Creating voice summary...")
    create_voice_summary()
    
    # Create temp directory for converted WAV files
    temp_dir = audio_dir / "temp_wav"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"\n🔄 Converting MP3-as-WAV files to proper WAV format...")
    
    converted_files = []
    successful_conversions = 0
    
    for i, file_path in enumerate(line_files):
        if file_path.stat().st_size > 1000:  # Skip empty files
            temp_wav = temp_dir / f"converted_{i:03d}.wav"
            print(f"  🔄 Converting {file_path.name}...")
            
            if convert_mp3_to_wav(file_path, temp_wav):
                converted_files.append(temp_wav)
                successful_conversions += 1
                print(f"  ✅ Converted line {i+1:03d}")
            else:
                print(f"  ❌ Failed to convert {file_path.name}")
        else:
            print(f"  ⚠️ Skipped empty file: {file_path.name}")
    
    print(f"\n✅ Successfully converted {successful_conversions}/{len(line_files)} files")
    
    if not converted_files:
        print("❌ No files were successfully converted")
        return
    
    # Concatenate the converted WAV files
    print(f"\n🎵 Concatenating {len(converted_files)} WAV files...")
    output_path = audio_dir / "chapter_0001_complete_with_awesome_zhou_mingrui.wav"
    
    if concatenate_wav_files(converted_files, output_path):
        # Get file info
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            
            print(f"\n🎉 SUCCESS! Created complete chapter audio:")
            print(f"   📁 File: {output_path.name}")
            print(f"   📊 Size: {file_size_mb:.2f} MB")
            print(f"   🌟 Features that awesome Zhou Mingrui voice you love!")
            
            # Clean up temp files
            for temp_file in converted_files:
                temp_file.unlink()
            temp_dir.rmdir()
            
            print(f"\n📁 Generated files:")
            print(f"   • {output_path.name} (full chapter with all voices)")
            print(f"   • chapter_0001_voice_summary.json (voice details)")
        else:
            print("❌ Output file was not created")
    else:
        print("❌ Failed to concatenate audio files")
        
        # Clean up temp files on failure
        for temp_file in converted_files:
            if temp_file.exists():
                temp_file.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()

if __name__ == "__main__":
    main()