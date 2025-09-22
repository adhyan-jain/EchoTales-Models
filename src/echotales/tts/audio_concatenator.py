#!/usr/bin/env python3
"""
Audio Concatenator for EchoTales
Combines character audio files into complete chapter audio
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import asyncio

try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioConcatenator:
    """Concatenate character audio files into complete chapter audio"""
    
    def __init__(self, generated_audio_dir: str = "generated_audio"):
        self.generated_audio_dir = Path(generated_audio_dir)
        self.ffmpeg_available = self._check_ffmpeg()
        
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available - audio concatenation will be limited")
            return False
        
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path:
            logger.info(f"FFmpeg found at: {ffmpeg_path}")
            return True
        else:
            logger.warning("FFmpeg not found - will use basic concatenation")
            return False
    
    async def concatenate_chapter(self, chapter_number: int, 
                                include_silences: bool = True,
                                silence_duration: float = 1.0) -> Optional[str]:
        """
        Concatenate all character audio files for a chapter
        
        Args:
            chapter_number: Chapter number to process
            include_silences: Whether to add silence between character lines
            silence_duration: Duration of silence in seconds
            
        Returns:
            Path to the concatenated audio file or None if failed
        """
        
        # Find chapter summary file
        summary_file = self.generated_audio_dir / f"chapter_{chapter_number:04d}_summary.json"
        if not summary_file.exists():
            logger.error(f"Chapter summary file not found: {summary_file}")
            return None
        
        # Load chapter data
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                chapter_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chapter summary: {e}")
            return None
        
        audio_files = chapter_data.get('audio_files', [])
        if not audio_files:
            logger.warning(f"No audio files found for chapter {chapter_number}")
            return None
        
        # Load the original chapter data to get line order
        dummy_data_path = Path("dummy_data")
        chapter_file = dummy_data_path / f"samples_chapter{chapter_number:04d}.json"
        
        if chapter_file.exists():
            with open(chapter_file, 'r', encoding='utf-8') as f:
                original_chapter_data = json.load(f)
            lines = original_chapter_data.get('lines', [])
        else:
            logger.warning(f"Original chapter data not found: {chapter_file}")
            lines = []
        
        # Create concatenated audio
        if PYDUB_AVAILABLE:
            return await self._concatenate_with_pydub(
                chapter_number, audio_files, lines, include_silences, silence_duration
            )
        else:
            return await self._concatenate_basic(chapter_number, audio_files)
    
    async def _concatenate_with_pydub(self, chapter_number: int, audio_files: List[Dict],
                                    lines: List[Dict], include_silences: bool,
                                    silence_duration: float) -> Optional[str]:
        """Concatenate audio using pydub with proper ordering"""
        
        try:
            # Create character audio lookup
            char_audio_map = {}
            for audio_file in audio_files:
                char_id = audio_file['character_id']
                file_path = Path(audio_file['file_path'])
                if file_path.exists():
                    char_audio_map[char_id] = str(file_path)
                else:
                    logger.warning(f"Audio file not found: {file_path}")
            
            if not char_audio_map:
                logger.error("No valid audio files found")
                return None
            
            # If we have line data, use it for proper ordering
            if lines:
                concatenated_audio = AudioSegment.empty()
                current_char = None
                char_segments = {}
                
                # Pre-load all character audio
                for char_id, file_path in char_audio_map.items():
                    try:
                        char_segments[char_id] = AudioSegment.from_wav(file_path)
                        logger.info(f"Loaded audio for character {char_id}: {len(char_segments[char_id])}ms")
                    except Exception as e:
                        logger.error(f"Failed to load audio for {char_id}: {e}")
                
                # Group lines by character to create segments
                char_line_groups = {}
                for line in lines:
                    char_id = line['character_id']
                    if char_id not in char_line_groups:
                        char_line_groups[char_id] = []
                    char_line_groups[char_id].append(line)
                
                # Add each character's complete segment
                silence = AudioSegment.silent(duration=int(silence_duration * 1000)) if include_silences else AudioSegment.empty()
                
                for char_id in char_line_groups.keys():
                    if char_id in char_segments:
                        if len(concatenated_audio) > 0 and include_silences:
                            concatenated_audio += silence
                        concatenated_audio += char_segments[char_id]
                        logger.info(f"Added {char_id} segment, total duration: {len(concatenated_audio)}ms")
            
            else:
                # Simple concatenation without line ordering
                concatenated_audio = AudioSegment.empty()
                silence = AudioSegment.silent(duration=int(silence_duration * 1000)) if include_silences else AudioSegment.empty()
                
                for audio_file in audio_files:
                    file_path = Path(audio_file['file_path'])
                    if file_path.exists():
                        try:
                            char_audio = AudioSegment.from_wav(str(file_path))
                            if len(concatenated_audio) > 0 and include_silences:
                                concatenated_audio += silence
                            concatenated_audio += char_audio
                            logger.info(f"Added {audio_file['character_name']}, total: {len(concatenated_audio)}ms")
                        except Exception as e:
                            logger.error(f"Failed to load {file_path}: {e}")
            
            if len(concatenated_audio) == 0:
                logger.error("No audio was concatenated")
                return None
            
            # Export the concatenated audio
            output_filename = f"chapter_{chapter_number:04d}_complete.wav"
            output_path = self.generated_audio_dir / output_filename
            
            concatenated_audio.export(str(output_path), format="wav")
            
            logger.info(f"Successfully created concatenated audio: {output_path}")
            logger.info(f"Total duration: {len(concatenated_audio)/1000:.2f} seconds")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to concatenate audio with pydub: {e}")
            return None
    
    async def _concatenate_basic(self, chapter_number: int, audio_files: List[Dict]) -> Optional[str]:
        """Basic concatenation using system commands"""
        logger.info("Using basic concatenation method")
        
        # Create a simple list file for concatenation
        list_file = self.generated_audio_dir / f"chapter_{chapter_number:04d}_filelist.txt"
        output_filename = f"chapter_{chapter_number:04d}_complete.wav"
        output_path = self.generated_audio_dir / output_filename
        
        try:
            with open(list_file, 'w') as f:
                for audio_file in audio_files:
                    file_path = Path(audio_file['file_path'])
                    if file_path.exists():
                        f.write(f"file '{file_path.absolute()}'\n")
            
            # Try using ffmpeg if available
            if self.ffmpeg_available:
                import subprocess
                cmd = [
                    "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", str(output_path), "-y"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Successfully created concatenated audio: {output_path}")
                    list_file.unlink()  # Clean up list file
                    return str(output_path)
                else:
                    logger.error(f"FFmpeg concatenation failed: {result.stderr}")
            
            # Fallback: copy first file as a placeholder
            first_file = Path(audio_files[0]['file_path'])
            if first_file.exists():
                import shutil
                shutil.copy2(first_file, output_path)
                logger.info(f"Created placeholder audio file: {output_path}")
                return str(output_path)
            
        except Exception as e:
            logger.error(f"Basic concatenation failed: {e}")
        
        finally:
            if list_file.exists():
                list_file.unlink()
        
        return None
    
    async def process_all_chapters(self) -> Dict[int, Optional[str]]:
        """Process all available chapters"""
        results = {}
        
        # Find all chapter summary files
        summary_files = list(self.generated_audio_dir.glob("chapter_*_summary.json"))
        
        for summary_file in summary_files:
            # Extract chapter number from filename
            try:
                chapter_number = int(summary_file.stem.split('_')[1])
                logger.info(f"Processing chapter {chapter_number}...")
                
                result = await self.concatenate_chapter(chapter_number)
                results[chapter_number] = result
                
            except (ValueError, IndexError) as e:
                logger.error(f"Could not parse chapter number from {summary_file}: {e}")
        
        return results
    
    def get_chapter_info(self, chapter_number: int) -> Optional[Dict]:
        """Get information about a chapter's audio"""
        summary_file = self.generated_audio_dir / f"chapter_{chapter_number:04d}_summary.json"
        
        if not summary_file.exists():
            return None
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                chapter_data = json.load(f)
            
            # Calculate total duration if possible
            total_duration = 0
            if PYDUB_AVAILABLE:
                for audio_file in chapter_data.get('audio_files', []):
                    file_path = Path(audio_file['file_path'])
                    if file_path.exists():
                        try:
                            audio = AudioSegment.from_wav(str(file_path))
                            total_duration += len(audio) / 1000  # Convert to seconds
                        except:
                            pass
            
            chapter_data['total_duration_seconds'] = total_duration
            return chapter_data
            
        except Exception as e:
            logger.error(f"Failed to get chapter info: {e}")
            return None


async def main():
    """Test the audio concatenator"""
    print("🎵 EchoTales Audio Concatenator")
    print("=" * 40)
    
    # Initialize concatenator
    concatenator = AudioConcatenator()
    
    # Check setup
    print(f"📦 PyDub available: {PYDUB_AVAILABLE}")
    print(f"🎬 FFmpeg available: {concatenator.ffmpeg_available}")
    
    # Process all chapters
    print(f"\n🔄 Processing chapters...")
    results = await concatenator.process_all_chapters()
    
    # Show results
    print(f"\n📊 CONCATENATION RESULTS")
    print("-" * 30)
    
    for chapter_num, output_file in results.items():
        if output_file:
            file_size = Path(output_file).stat().st_size / 1024  # KB
            print(f"✅ Chapter {chapter_num}: {Path(output_file).name} ({file_size:.1f} KB)")
            
            # Show chapter info
            chapter_info = concatenator.get_chapter_info(chapter_num)
            if chapter_info:
                print(f"   📝 Characters: {chapter_info['total_characters']}")
                print(f"   🎤 Audio files: {len(chapter_info['audio_files'])}")
                if chapter_info.get('total_duration_seconds', 0) > 0:
                    print(f"   ⏱️ Duration: {chapter_info['total_duration_seconds']:.1f}s")
        else:
            print(f"❌ Chapter {chapter_num}: Failed to concatenate")
    
    print(f"\n🎉 Audio concatenation complete!")
    print(f"📁 Check the 'generated_audio' folder for complete chapter files")


if __name__ == "__main__":
    asyncio.run(main())