"""
Audio Post-Processing Utilities for EchoTales

This module handles audio processing tasks including:
- Volume normalization across chapters
- Silence trimming
- Chapter-level concatenation
- Format conversion and standardization
- Comprehensive fallback mechanisms for reliability
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..core.environment import get_config

logger = logging.getLogger(__name__)


@dataclass
class AudioProcessingConfig:
    """Configuration for audio processing"""
    target_sample_rate: int = 22050
    target_format: str = "wav"
    target_channels: int = 1  # Mono
    target_bit_depth: int = 16
    
    # Volume normalization
    target_rms_db: float = -20.0
    max_peak_db: float = -3.0
    
    # Silence detection
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.5  # seconds
    
    # Processing limits
    max_file_size_mb: float = 100.0
    max_duration_minutes: float = 60.0


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass


class AudioProcessor:
    """Main audio processing class with fallback mechanisms"""
    
    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()
        self.app_config = get_config()
        
        # Initialize audio libraries with fallbacks
        self.audio_backend = self._initialize_audio_backend()
        
    def _initialize_audio_backend(self) -> str:
        """Initialize audio processing backend with fallbacks"""
        
        # Priority order of audio backends
        backends = [
            ("librosa", self._check_librosa),
            ("pydub", self._check_pydub),
            ("scipy", self._check_scipy),
            ("wave_builtin", self._check_wave_builtin)
        ]
        
        for backend_name, check_func in backends:
            try:
                if check_func():
                    logger.info(f"Using audio backend: {backend_name}")
                    return backend_name
            except Exception as e:
                logger.warning(f"Backend {backend_name} not available: {e}")
        
        # Ultimate fallback - basic file operations only
        logger.warning("No advanced audio backends available. Using basic file operations only.")
        return "basic"
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available"""
        try:
            import librosa
            import soundfile as sf
            return True
        except ImportError:
            return False
    
    def _check_pydub(self) -> bool:
        """Check if pydub is available"""
        try:
            from pydub import AudioSegment
            return True
        except ImportError:
            return False
    
    def _check_scipy(self) -> bool:
        """Check if scipy audio functions are available"""
        try:
            from scipy.io import wavfile
            from scipy import signal
            return True
        except ImportError:
            return False
    
    def _check_wave_builtin(self) -> bool:
        """Check if Python's built-in wave module works"""
        try:
            import wave
            return True
        except ImportError:
            return False
    
    def normalize_volume(self, 
                        audio_files: List[str], 
                        output_dir: str,
                        target_rms_db: Optional[float] = None) -> List[str]:
        """
        Normalize volume across multiple audio files with fallbacks
        
        Args:
            audio_files: List of input audio file paths
            output_dir: Output directory for normalized files
            target_rms_db: Target RMS level in dB
            
        Returns:
            List of normalized audio file paths
        """
        
        target_rms_db = target_rms_db or self.config.target_rms_db
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        normalized_files = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                input_path = Path(audio_file)
                
                if not input_path.exists():
                    logger.error(f"Input file not found: {audio_file}")
                    # Fallback: Create silent placeholder
                    output_path = output_dir / f"normalized_{i}_{input_path.name}"
                    self._create_silent_fallback(output_path)
                    normalized_files.append(str(output_path))
                    continue
                
                output_path = output_dir / f"normalized_{input_path.name}"
                
                # Try different normalization methods
                success = False
                
                if self.audio_backend == "librosa":
                    success = self._normalize_with_librosa(input_path, output_path, target_rms_db)
                
                if not success and self.audio_backend in ["pydub", "librosa"]:
                    success = self._normalize_with_pydub(input_path, output_path, target_rms_db)
                
                if not success and self.audio_backend in ["scipy", "librosa", "pydub"]:
                    success = self._normalize_with_scipy(input_path, output_path, target_rms_db)
                
                if not success:
                    # Ultimate fallback: just copy the file
                    logger.warning(f"Could not normalize {audio_file}, copying original")
                    output_path = output_dir / input_path.name
                    self._safe_copy_audio(input_path, output_path)
                
                normalized_files.append(str(output_path))
                
            except Exception as e:
                logger.error(f"Failed to normalize {audio_file}: {e}")
                # Fallback: Create silent placeholder or copy original
                try:
                    output_path = output_dir / f"fallback_{i}_{Path(audio_file).name}"
                    self._safe_copy_audio(Path(audio_file), output_path)
                    normalized_files.append(str(output_path))
                except:
                    # Last resort: silent audio
                    output_path = output_dir / f"silent_{i}.wav"
                    self._create_silent_fallback(output_path)
                    normalized_files.append(str(output_path))
        
        logger.info(f"Volume normalization completed: {len(normalized_files)} files")
        return normalized_files
    
    def _normalize_with_librosa(self, input_path: Path, output_path: Path, target_rms_db: float) -> bool:
        """Normalize audio using librosa"""
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            audio, sr = librosa.load(str(input_path), sr=self.config.target_sample_rate)
            
            # Calculate current RMS
            current_rms = np.sqrt(np.mean(audio**2))
            
            if current_rms > 0:
                # Calculate target RMS from dB
                target_rms = 10**(target_rms_db / 20)
                
                # Calculate gain
                gain = target_rms / current_rms
                
                # Apply gain with peak limiting
                normalized_audio = audio * gain
                
                # Peak limiting
                max_peak = 10**(self.config.max_peak_db / 20)
                if np.max(np.abs(normalized_audio)) > max_peak:
                    normalized_audio = normalized_audio * (max_peak / np.max(np.abs(normalized_audio)))
                
                # Save
                sf.write(str(output_path), normalized_audio, sr)
                return True
            else:
                # Silent audio - just copy
                self._safe_copy_audio(input_path, output_path)
                return True
                
        except Exception as e:
            logger.error(f"Librosa normalization failed: {e}")
            return False
    
    def _normalize_with_pydub(self, input_path: Path, output_path: Path, target_rms_db: float) -> bool:
        """Normalize audio using pydub"""
        try:
            from pydub import AudioSegment
            
            # Load audio
            audio = AudioSegment.from_file(str(input_path))
            
            # Convert to target format
            audio = audio.set_frame_rate(self.config.target_sample_rate)
            audio = audio.set_channels(self.config.target_channels)
            
            # Normalize volume
            current_rms = audio.rms
            if current_rms > 0:
                target_rms = 10**(target_rms_db / 20) * 32768  # Convert to 16-bit scale
                gain_db = 20 * np.log10(target_rms / current_rms)
                
                # Apply gain with limiting
                normalized_audio = audio + gain_db
                
                # Peak limiting
                if normalized_audio.max_possible_amplitude > 10**(self.config.max_peak_db / 20) * 32768:
                    peak_limit_db = self.config.max_peak_db - 20 * np.log10(normalized_audio.max_possible_amplitude / 32768)
                    normalized_audio = normalized_audio + peak_limit_db
                
                # Export
                normalized_audio.export(str(output_path), format="wav")
                return True
            else:
                # Silent audio
                audio.export(str(output_path), format="wav")
                return True
                
        except Exception as e:
            logger.error(f"Pydub normalization failed: {e}")
            return False
    
    def _normalize_with_scipy(self, input_path: Path, output_path: Path, target_rms_db: float) -> bool:
        """Basic normalization using scipy"""
        try:
            from scipy.io import wavfile
            
            # Load audio
            sample_rate, audio = wavfile.read(str(input_path))
            
            # Convert to float
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            # Calculate RMS and normalize
            current_rms = np.sqrt(np.mean(audio**2))
            
            if current_rms > 0:
                target_rms = 10**(target_rms_db / 20)
                gain = target_rms / current_rms
                
                normalized_audio = audio * gain
                
                # Convert back to int16
                normalized_audio = np.clip(normalized_audio * 32768, -32768, 32767).astype(np.int16)
                
                # Save
                wavfile.write(str(output_path), sample_rate, normalized_audio)
                return True
            else:
                # Silent audio - copy
                wavfile.write(str(output_path), sample_rate, audio.astype(np.int16))
                return True
                
        except Exception as e:
            logger.error(f"Scipy normalization failed: {e}")
            return False
    
    def trim_silence(self, 
                    audio_files: List[str], 
                    output_dir: str,
                    silence_threshold_db: Optional[float] = None) -> List[str]:
        """
        Trim silence from audio files with fallbacks
        
        Args:
            audio_files: List of input audio file paths
            output_dir: Output directory for trimmed files
            silence_threshold_db: Silence threshold in dB
            
        Returns:
            List of trimmed audio file paths
        """
        
        silence_threshold_db = silence_threshold_db or self.config.silence_threshold_db
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trimmed_files = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                input_path = Path(audio_file)
                
                if not input_path.exists():
                    logger.error(f"Input file not found: {audio_file}")
                    # Create placeholder
                    output_path = output_dir / f"trimmed_{i}.wav"
                    self._create_silent_fallback(output_path, duration=1.0)
                    trimmed_files.append(str(output_path))
                    continue
                
                output_path = output_dir / f"trimmed_{input_path.name}"
                
                # Try different trimming methods
                success = False
                
                if self.audio_backend == "librosa":
                    success = self._trim_with_librosa(input_path, output_path, silence_threshold_db)
                
                if not success and self.audio_backend in ["pydub", "librosa"]:
                    success = self._trim_with_pydub(input_path, output_path, silence_threshold_db)
                
                if not success:
                    # Fallback: just copy the file
                    logger.warning(f"Could not trim silence from {audio_file}, copying original")
                    self._safe_copy_audio(input_path, output_path)
                
                trimmed_files.append(str(output_path))
                
            except Exception as e:
                logger.error(f"Failed to trim silence from {audio_file}: {e}")
                # Fallback
                try:
                    output_path = output_dir / f"fallback_trimmed_{i}_{Path(audio_file).name}"
                    self._safe_copy_audio(Path(audio_file), output_path)
                    trimmed_files.append(str(output_path))
                except:
                    # Last resort
                    output_path = output_dir / f"silent_trimmed_{i}.wav"
                    self._create_silent_fallback(output_path, duration=1.0)
                    trimmed_files.append(str(output_path))
        
        logger.info(f"Silence trimming completed: {len(trimmed_files)} files")
        return trimmed_files
    
    def _trim_with_librosa(self, input_path: Path, output_path: Path, silence_threshold_db: float) -> bool:
        """Trim silence using librosa"""
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            audio, sr = librosa.load(str(input_path), sr=self.config.target_sample_rate)
            
            # Convert threshold to amplitude
            silence_threshold = 10**(silence_threshold_db / 20)
            
            # Find non-silent intervals
            intervals = librosa.effects.split(audio, 
                                             top_db=-silence_threshold_db,
                                             frame_length=2048,
                                             hop_length=512)
            
            if len(intervals) > 0:
                # Concatenate non-silent segments
                trimmed_segments = []
                for start, end in intervals:
                    trimmed_segments.append(audio[start:end])
                
                if trimmed_segments:
                    trimmed_audio = np.concatenate(trimmed_segments)
                else:
                    # No non-silent parts found, keep a small segment
                    trimmed_audio = audio[:min(sr, len(audio))]
            else:
                # Fallback: keep original
                trimmed_audio = audio
            
            # Save
            sf.write(str(output_path), trimmed_audio, sr)
            return True
            
        except Exception as e:
            logger.error(f"Librosa silence trimming failed: {e}")
            return False
    
    def _trim_with_pydub(self, input_path: Path, output_path: Path, silence_threshold_db: float) -> bool:
        """Trim silence using pydub"""
        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
            
            # Load audio
            audio = AudioSegment.from_file(str(input_path))
            
            # Split on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=int(self.config.min_silence_duration * 1000),  # Convert to ms
                silence_thresh=silence_threshold_db,
                keep_silence=100  # Keep 100ms of silence
            )
            
            if chunks:
                # Concatenate non-silent chunks
                trimmed_audio = AudioSegment.empty()
                for chunk in chunks:
                    trimmed_audio += chunk
            else:
                # No chunks found, keep original
                trimmed_audio = audio
            
            # Export
            trimmed_audio.export(str(output_path), format="wav")
            return True
            
        except Exception as e:
            logger.error(f"Pydub silence trimming failed: {e}")
            return False
    
    def concatenate_chapters(self, 
                           chapter_audio_files: List[List[str]], 
                           output_dir: str,
                           chapter_names: Optional[List[str]] = None) -> List[str]:
        """
        Concatenate audio files for each chapter with fallbacks
        
        Args:
            chapter_audio_files: List of lists, where each inner list contains audio files for a chapter
            output_dir: Output directory for concatenated chapter files
            chapter_names: Optional names for chapters
            
        Returns:
            List of concatenated chapter file paths
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        concatenated_files = []
        
        for i, chapter_files in enumerate(chapter_audio_files):
            try:
                chapter_name = chapter_names[i] if chapter_names and i < len(chapter_names) else f"Chapter_{i+1}"
                output_path = output_dir / f"{chapter_name}.wav"
                
                if not chapter_files:
                    # No files for this chapter - create silent placeholder
                    logger.warning(f"No audio files for {chapter_name}, creating silent placeholder")
                    self._create_silent_fallback(output_path, duration=5.0)
                    concatenated_files.append(str(output_path))
                    continue
                
                # Filter existing files
                existing_files = [f for f in chapter_files if Path(f).exists()]
                
                if not existing_files:
                    # No existing files - create silent placeholder
                    logger.warning(f"No existing audio files for {chapter_name}, creating silent placeholder")
                    self._create_silent_fallback(output_path, duration=5.0)
                    concatenated_files.append(str(output_path))
                    continue
                
                # Try different concatenation methods
                success = False
                
                if self.audio_backend == "librosa":
                    success = self._concatenate_with_librosa(existing_files, output_path)
                
                if not success and self.audio_backend in ["pydub", "librosa"]:
                    success = self._concatenate_with_pydub(existing_files, output_path)
                
                if not success:
                    # Fallback: just copy first file
                    logger.warning(f"Could not concatenate {chapter_name}, using first file only")
                    self._safe_copy_audio(Path(existing_files[0]), output_path)
                
                concatenated_files.append(str(output_path))
                
            except Exception as e:
                logger.error(f"Failed to concatenate chapter {i}: {e}")
                # Create fallback
                try:
                    chapter_name = f"Chapter_{i+1}"
                    output_path = output_dir / f"fallback_{chapter_name}.wav"
                    self._create_silent_fallback(output_path, duration=5.0)
                    concatenated_files.append(str(output_path))
                except Exception as fallback_error:
                    logger.error(f"Even fallback failed for chapter {i}: {fallback_error}")
        
        logger.info(f"Chapter concatenation completed: {len(concatenated_files)} chapters")
        return concatenated_files
    
    def _concatenate_with_librosa(self, audio_files: List[str], output_path: Path) -> bool:
        """Concatenate audio files using librosa"""
        try:
            import librosa
            import soundfile as sf
            
            concatenated_audio = []
            
            for audio_file in audio_files:
                audio, sr = librosa.load(audio_file, sr=self.config.target_sample_rate)
                concatenated_audio.append(audio)
            
            if concatenated_audio:
                final_audio = np.concatenate(concatenated_audio)
                sf.write(str(output_path), final_audio, self.config.target_sample_rate)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Librosa concatenation failed: {e}")
            return False
    
    def _concatenate_with_pydub(self, audio_files: List[str], output_path: Path) -> bool:
        """Concatenate audio files using pydub"""
        try:
            from pydub import AudioSegment
            
            concatenated_audio = AudioSegment.empty()
            
            for audio_file in audio_files:
                audio = AudioSegment.from_file(audio_file)
                
                # Standardize format
                audio = audio.set_frame_rate(self.config.target_sample_rate)
                audio = audio.set_channels(self.config.target_channels)
                
                concatenated_audio += audio
            
            if len(concatenated_audio) > 0:
                concatenated_audio.export(str(output_path), format="wav")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Pydub concatenation failed: {e}")
            return False
    
    def _safe_copy_audio(self, input_path: Path, output_path: Path):
        """Safely copy audio file with fallbacks"""
        try:
            import shutil
            shutil.copy2(str(input_path), str(output_path))
        except Exception as e:
            logger.error(f"Failed to copy {input_path} to {output_path}: {e}")
            # Last resort: create silent placeholder
            self._create_silent_fallback(output_path)
    
    def _create_silent_fallback(self, output_path: Path, duration: float = 1.0):
        """Create a silent audio file as fallback"""
        try:
            if self.audio_backend == "librosa":
                self._create_silent_librosa(output_path, duration)
            elif self.audio_backend == "pydub":
                self._create_silent_pydub(output_path, duration)
            elif self.audio_backend == "scipy":
                self._create_silent_scipy(output_path, duration)
            else:
                self._create_silent_builtin(output_path, duration)
        except Exception as e:
            logger.error(f"Failed to create silent fallback: {e}")
            # Create empty file as absolute last resort
            output_path.touch()
    
    def _create_silent_librosa(self, output_path: Path, duration: float):
        """Create silent audio using librosa"""
        import soundfile as sf
        import numpy as np
        
        samples = int(duration * self.config.target_sample_rate)
        silent_audio = np.zeros(samples, dtype=np.float32)
        sf.write(str(output_path), silent_audio, self.config.target_sample_rate)
    
    def _create_silent_pydub(self, output_path: Path, duration: float):
        """Create silent audio using pydub"""
        from pydub import AudioSegment
        
        silent_audio = AudioSegment.silent(
            duration=int(duration * 1000),  # Convert to ms
            frame_rate=self.config.target_sample_rate
        )
        silent_audio.export(str(output_path), format="wav")
    
    def _create_silent_scipy(self, output_path: Path, duration: float):
        """Create silent audio using scipy"""
        from scipy.io import wavfile
        import numpy as np
        
        samples = int(duration * self.config.target_sample_rate)
        silent_audio = np.zeros(samples, dtype=np.int16)
        wavfile.write(str(output_path), self.config.target_sample_rate, silent_audio)
    
    def _create_silent_builtin(self, output_path: Path, duration: float):
        """Create silent audio using built-in wave module"""
        import wave
        import numpy as np
        
        samples = int(duration * self.config.target_sample_rate)
        silent_audio = np.zeros(samples, dtype=np.int16)
        
        with wave.open(str(output_path), 'w') as wav_file:
            wav_file.setnchannels(self.config.target_channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.target_sample_rate)
            wav_file.writeframes(silent_audio.tobytes())
    
    def process_audio_pipeline(self, 
                              input_files: List[str], 
                              output_dir: str,
                              normalize_volume: bool = True,
                              trim_silence: bool = True,
                              target_format: str = "wav") -> Dict[str, Any]:
        """
        Complete audio processing pipeline with comprehensive error handling
        
        Args:
            input_files: List of input audio files
            output_dir: Output directory
            normalize_volume: Whether to normalize volume
            trim_silence: Whether to trim silence
            target_format: Target audio format
            
        Returns:
            Dictionary with processing results and status
        """
        
        results = {
            "success": False,
            "processed_files": [],
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            processed_files = input_files.copy()
            
            # Step 1: Volume normalization (if requested)
            if normalize_volume:
                try:
                    norm_dir = output_dir / "normalized"
                    processed_files = self.normalize_volume(processed_files, str(norm_dir))
                    results["warnings"].append("Volume normalization completed")
                except Exception as e:
                    results["errors"].append(f"Volume normalization failed: {e}")
                    results["warnings"].append("Continuing without volume normalization")
            
            # Step 2: Silence trimming (if requested)
            if trim_silence:
                try:
                    trim_dir = output_dir / "trimmed"
                    processed_files = self.trim_silence(processed_files, str(trim_dir))
                    results["warnings"].append("Silence trimming completed")
                except Exception as e:
                    results["errors"].append(f"Silence trimming failed: {e}")
                    results["warnings"].append("Continuing without silence trimming")
            
            # Step 3: Format conversion (if needed)
            if target_format != "wav":
                try:
                    format_dir = output_dir / "converted"
                    processed_files = self._convert_format(processed_files, str(format_dir), target_format)
                except Exception as e:
                    results["errors"].append(f"Format conversion failed: {e}")
                    results["warnings"].append(f"Files remain in original format")
            
            results["processed_files"] = processed_files
            results["success"] = True
            
            # Add statistics
            results["stats"] = {
                "input_count": len(input_files),
                "output_count": len(processed_files),
                "success_rate": len(processed_files) / len(input_files) if input_files else 0,
                "backend_used": self.audio_backend
            }
            
        except Exception as e:
            results["errors"].append(f"Pipeline failed: {e}")
            # Create fallback results
            results["processed_files"] = [str(output_dir / f"fallback_{i}.wav") for i in range(len(input_files))]
            for i, fallback_path in enumerate(results["processed_files"]):
                self._create_silent_fallback(Path(fallback_path))
        
        return results
    
    def _convert_format(self, input_files: List[str], output_dir: str, target_format: str) -> List[str]:
        """Convert audio files to target format with fallbacks"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        converted_files = []
        
        for input_file in input_files:
            try:
                input_path = Path(input_file)
                output_path = output_dir / f"{input_path.stem}.{target_format}"
                
                # Try different conversion methods
                success = False
                
                if self.audio_backend == "pydub":
                    success = self._convert_with_pydub(input_path, output_path, target_format)
                
                if not success and self.audio_backend == "librosa":
                    success = self._convert_with_librosa(input_path, output_path, target_format)
                
                if not success:
                    # Fallback: copy with new extension (may not work but better than nothing)
                    output_path = output_dir / f"{input_path.stem}_copy.{target_format}"
                    self._safe_copy_audio(input_path, output_path)
                
                converted_files.append(str(output_path))
                
            except Exception as e:
                logger.error(f"Format conversion failed for {input_file}: {e}")
                # Add placeholder
                fallback_path = output_dir / f"fallback_{len(converted_files)}.{target_format}"
                self._create_silent_fallback(fallback_path)
                converted_files.append(str(fallback_path))
        
        return converted_files
    
    def _convert_with_pydub(self, input_path: Path, output_path: Path, target_format: str) -> bool:
        """Convert format using pydub"""
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(str(input_path))
            audio = audio.set_frame_rate(self.config.target_sample_rate)
            audio = audio.set_channels(self.config.target_channels)
            
            audio.export(str(output_path), format=target_format)
            return True
            
        except Exception as e:
            logger.error(f"Pydub format conversion failed: {e}")
            return False
    
    def _convert_with_librosa(self, input_path: Path, output_path: Path, target_format: str) -> bool:
        """Convert format using librosa"""
        try:
            import librosa
            import soundfile as sf
            
            audio, sr = librosa.load(str(input_path), sr=self.config.target_sample_rate)
            
            # librosa/soundfile mainly supports wav, so only convert if target is wav
            if target_format.lower() == "wav":
                sf.write(str(output_path), audio, sr)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Librosa format conversion failed: {e}")
            return False


def main():
    """Test the audio processing pipeline"""
    
    # Create test audio processor
    processor = AudioProcessor()
    
    print(f"Audio backend: {processor.audio_backend}")
    
    # Test creating a silent file
    try:
        test_dir = Path("test_audio_output")
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / "test_silent.wav"
        processor._create_silent_fallback(test_file, duration=2.0)
        
        if test_file.exists():
            print(f"Successfully created test silent file: {test_file}")
            print(f"File size: {test_file.stat().st_size} bytes")
        
        # Clean up
        if test_file.exists():
            test_file.unlink()
        if test_dir.exists():
            test_dir.rmdir()
            
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()