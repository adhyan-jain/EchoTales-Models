#!/usr/bin/env python3
"""
BookNLP Microservice for EchoTales
Handles book text processing via REST API
"""

import os
import sys
import time
import tempfile
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from glob import glob
import subprocess

# Import the output checker utility
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.booknlp_output_checker import BookNLPOutputChecker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="EchoTales BookNLP Service",
    description="BookNLP processing microservice for literary text analysis",
    version="1.0.0"
)

# Global variables
booknlp_model = None
output_base_dir = Path("C:/Users/Adhyan/OneDrive/Desktop/EchoTales-Enhanced/modelsbooknlp/output")
output_base_dir.mkdir(exist_ok=True)
output_checker = None

class BookProcessingRequest(BaseModel):
    text: str
    book_id: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    pipeline: Optional[str] = "entity,quote,supersense,event,coref"
    skip_if_exists: Optional[bool] = True
    
class BookProcessingResponse(BaseModel):
    status: str
    message: str
    book_id: str
    processing_time: float
    output_directory: str
    files_generated: List[str]
    entities_count: Optional[int] = None
    quotes_count: Optional[int] = None
    characters_count: Optional[int] = None
    skipped_processing: Optional[bool] = None
    # Extended fields
    chapters_count: Optional[int] = None
    parent_json_path: Optional[str] = None
    parent_json: Optional[Dict] = None
    dialogue_files: Optional[List[str]] = None
    dialogue_summary: Optional[Dict] = None
    # External command outputs
    commands_output: Optional[str] = None
    py_compile_ok: Optional[bool] = None
    dialogue_cmd_ok: Optional[bool] = None

@app.on_event("startup")
async def startup_event():
    """Initialize BookNLP on startup"""
    global booknlp_model, output_checker
    
    try:
        # Import BookNLP
        from booknlp.booknlp import BookNLP
        
        # Model parameters - using the same configuration from your test files
        model_params = {
            "pipeline": "entity,quote,supersense,event,coref",
            "model": "custom",
            "entity_model_path": r"C:\Users\Adhyan\booknlp_models\entities_google_bert_uncased_L-6_H-768_A-12-v1.0_modified.model",
            "coref_model_path": r"C:\Users\Adhyan\booknlp_models\coref_google_bert_uncased_L-12_H-768_A-12-v1.0_modified.model",
            "quote_attribution_model_path": r"C:\Users\Adhyan\booknlp_models\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1_modified.model",
            "bert_model_path": r"C:\Users\Adhyan\.cache\huggingface\hub"
        }
        
        # Verify model files exist
        for key, path in model_params.items():
            if key.endswith('_path') and key != 'bert_model_path':
                if not os.path.exists(path):
                    logger.error(f"Model file not found: {path}")
                    raise FileNotFoundError(f"Model file not found: {path}")
                else:
                    logger.info(f"✓ Model file found: {os.path.basename(path)}")
        
        # Initialize BookNLP
        booknlp_model = BookNLP("en", model_params)
        
        # Initialize output checker
        output_checker = BookNLPOutputChecker(output_base_dir=str(output_base_dir))
        
        logger.info("BookNLP service started successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import BookNLP: {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to initialize BookNLP: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint with service status"""
    return {
        "service": "EchoTales BookNLP Service",
        "status": "running",
        "version": "1.0.0",
        "booknlp_available": booknlp_model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if booknlp_model else "unhealthy",
        "booknlp_loaded": booknlp_model is not None,
        "timestamp": time.time()
    }

@app.post("/process", response_model=BookProcessingResponse)
async def process_book_text(request: BookProcessingRequest):
    """Process book text using BookNLP"""
    
    if not booknlp_model:
        raise HTTPException(status_code=503, detail="BookNLP service not available")
    
    if not output_checker:
        raise HTTPException(status_code=503, detail="Output checker not available")
    
    start_time = time.time()
    
    try:
        # Generate book ID if not provided
        if not request.book_id:
            if request.title:
                request.book_id = output_checker.generate_book_id(request.text, request.title)
            else:
                text_hash = hashlib.md5(request.text.encode()).hexdigest()[:12]
                request.book_id = f"book_{text_hash}"
        
        logger.info(f"Processing book: {request.book_id}")
        logger.info(f"Text length: {len(request.text)} characters")
        
        # Check if output already exists and should be skipped
        if request.skip_if_exists:
            existing_results = output_checker.get_existing_booknlp_results(request.book_id)
            if existing_results:
                processing_time = time.time() - start_time
                logger.info(f"⏩ Skipping processing for {request.book_id} - output already exists")
                
                return BookProcessingResponse(
                    status=existing_results["status"],
                    message=existing_results["message"],
                    book_id=existing_results["book_id"],
                    processing_time=processing_time,
                    output_directory=existing_results["output_directory"],
                    files_generated=existing_results["files_generated"],
                    entities_count=existing_results["entities_count"],
                    quotes_count=existing_results["quotes_count"],
                    characters_count=existing_results["characters_count"],
                    skipped_processing=True
                )
        
        # Create temporary file for input text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(request.text)
            temp_input_path = temp_file.name
        
        try:
            # Create output directory for this book
            book_output_dir = output_base_dir / request.book_id
            book_output_dir.mkdir(exist_ok=True)
            
            # Process the book
            logger.info("Starting BookNLP processing...")
            # Use positional args to be compatible across BookNLP versions
            booknlp_model.process(
                temp_input_path,
                str(book_output_dir),
                request.book_id
            )
            
            # Run dialogue processing and build parent JSON
            dialogue_summary = None
            dialogue_files: List[str] = []
            parent_json_path: Optional[str] = None
            chapters_count: Optional[int] = None

            try:
                # Defer import to runtime
                from dialogue_processor import FixedDialogueProcessor  # type: ignore
                processor = FixedDialogueProcessor(
                    output_dir=str(book_output_dir),
                    dialogue_dir=str(Path("data/processed/dialogues"))
                )
                # Use the processed book_id for dialogue processing
                dialogue_summary, dialogue_files = processor.process_book_dialogue(request.book_id)
                logger.info(
                    f"Processing complete! Generated {len(dialogue_files)} chapter files"
                )

                # Build parent JSON including chapters array and advanced characters
                parent_json_path, chapters_count = build_parent_dialogue_json(
                    request.book_id, dialogue_files
                )
            except Exception as dp_err:
                logger.warning(f"Dialogue processing or parent JSON build failed: {dp_err}")
            
            # Run external commands as requested and capture outputs
            commands_output_parts: List[str] = []
            py_compile_ok: Optional[bool] = None
            dialogue_cmd_ok: Optional[bool] = None

            try:
                project_root = Path(__file__).resolve().parents[2]
                # 1) python -m py_compile advanced_character_processor.py
                logger.info("Running py_compile for advanced_character_processor.py...")
                pyc = subprocess.run(
                    [sys.executable, "-m", "py_compile", "advanced_character_processor.py"],
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    shell=False
                )
                py_compile_ok = pyc.returncode == 0
                commands_output_parts.append("[py_compile stdout]\n" + (pyc.stdout or ""))
                if pyc.stderr:
                    commands_output_parts.append("[py_compile stderr]\n" + pyc.stderr)
            except Exception as e:
                logger.warning(f"py_compile failed: {e}")
                commands_output_parts.append(f"[py_compile exception]\n{e}\n")
                py_compile_ok = False

            try:
                # 2) python -c "from dialogue_processor import FixedDialogueProcessor; ..."
                book_id_for_cmd = request.book_id or "lord_of_mysteries"
                book_dir_for_cmd = str(book_output_dir).replace('\\', r'\\')
                dlg_dir_for_cmd = str(Path("data/processed/dialogues")).replace('\\', r'\\')
                code = (
                    "from dialogue_processor import FixedDialogueProcessor; "
                    f"processor = FixedDialogueProcessor(output_dir=\"{book_dir_for_cmd}\", dialogue_dir=\"{dlg_dir_for_cmd}\"); "
                    f"summary, files = processor.process_book_dialogue('{book_id_for_cmd}'); "
                    "print(f'Processing complete! Generated {len(files)} chapter files')"
                )
                logger.info("Running dialogue processing via python -c ...")
                dlg = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=str(project_root),
                    capture_output=True,
                    text=True,
                    shell=False
                )
                dialogue_cmd_ok = dlg.returncode == 0
                commands_output_parts.append("[dialogue_cmd stdout]\n" + (dlg.stdout or ""))
                if dlg.stderr:
                    commands_output_parts.append("[dialogue_cmd stderr]\n" + dlg.stderr)
            except Exception as e:
                logger.warning(f"dialogue python -c failed: {e}")
                commands_output_parts.append(f"[dialogue_cmd exception]\n{e}\n")
                dialogue_cmd_ok = False

            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"BookNLP processing completed in {processing_time:.2f} seconds")
            
            # Get list of generated files
            generated_files = []
            if book_output_dir.exists():
                generated_files = [f.name for f in book_output_dir.iterdir() if f.is_file()]
            
            # Try to extract some statistics from the output
            entities_count, quotes_count, characters_count = extract_statistics(book_output_dir)
            
            # Load parent JSON content if available
            parent_json_content: Optional[Dict] = None
            if parent_json_path:
                try:
                    with open(parent_json_path, 'r', encoding='utf-8') as pf:
                        parent_json_content = json.load(pf)
                except Exception as e:
                    logger.warning(f"Failed to load parent JSON content: {e}")

            return BookProcessingResponse(
                status="success",
                message="Book processing completed successfully",
                book_id=request.book_id,
                processing_time=processing_time,
                output_directory=str(book_output_dir),
                files_generated=generated_files,
                entities_count=entities_count,
                quotes_count=quotes_count,
                characters_count=characters_count,
                skipped_processing=False,
                chapters_count=chapters_count,
                parent_json_path=parent_json_path,
                parent_json=parent_json_content,
                dialogue_files=dialogue_files or None,
                dialogue_summary=dialogue_summary or None
                ,
                commands_output="\n".join(commands_output_parts) if commands_output_parts else None,
                py_compile_ok=py_compile_ok,
                dialogue_cmd_ok=dialogue_cmd_ok
            )
            
        finally:
            # Clean up temporary input file
            try:
                os.unlink(temp_input_path)
            except OSError:
                pass
                
    except Exception as e:
        logger.error(f"BookNLP processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

def extract_statistics(output_dir: Path) -> tuple:
    """Extract basic statistics from BookNLP output files"""
    entities_count = None
    quotes_count = None
    characters_count = None
    
    try:
        # Try to read entities file
        entities_file = output_dir / f"{output_dir.name}.entities"
        if entities_file.exists():
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_count = sum(1 for line in f) - 1  # Subtract header
        
        # Try to read quotes file
        quotes_file = output_dir / f"{output_dir.name}.quotes"
        if quotes_file.exists():
            with open(quotes_file, 'r', encoding='utf-8') as f:
                quotes_count = sum(1 for line in f) - 1  # Subtract header
        
        # Try to read book file for character count
        book_file = output_dir / f"{output_dir.name}.book"
        if book_file.exists():
            try:
                with open(book_file, 'r', encoding='utf-8') as f:
                    book_data = json.load(f)
                    if 'characters' in book_data:
                        characters_count = len(book_data['characters'])
            except json.JSONDecodeError:
                pass
        
    except Exception as e:
        logger.warning(f"Failed to extract statistics: {e}")
    
    return entities_count, quotes_count, characters_count

def build_parent_dialogue_json(book_id: str, chapter_files: List[str]) -> Tuple[str, int]:
    """Aggregate per-chapter dialogue JSONs into a single parent JSON and include advanced characters.

    Returns tuple of (parent_json_path, chapters_count).
    """
    try:
        # Ensure we have chapter file paths; if only names are provided, expand from dialogue dir
        dialogue_dir = Path("data/processed/dialogues")
        resolved_files: List[Path] = []
        for f in chapter_files:
            p = Path(f)
            if not p.is_absolute():
                p = dialogue_dir / p
            if p.exists():
                resolved_files.append(p)

        # Fallback: discover by glob if list empty
        if not resolved_files:
            pattern = str(dialogue_dir / f"{book_id}_chapter*.json")
            resolved_files = [Path(p) for p in glob(pattern)]

        chapters: List[Dict] = []
        for p in sorted(resolved_files):
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    chapters.append(json.load(fh))
            except Exception as e:
                logger.warning(f"Failed to load chapter file {p}: {e}")

        # Load advanced characters if available
        advanced_characters: Optional[Dict] = None
        adv_paths = [
            Path("data/processed/characters/advanced_characters.json"),
            Path("characters/advanced_characters.json")
        ]
        for ap in adv_paths:
            if ap.exists():
                try:
                    with open(ap, 'r', encoding='utf-8') as fh:
                        advanced_characters = json.load(fh)
                    break
                except Exception as e:
                    logger.warning(f"Failed to read advanced characters from {ap}: {e}")

        # Build parent JSON in requested format: characters + chapters
        characters_list: List[Dict] = []
        if isinstance(advanced_characters, dict):
            # Most files store characters under 'characters'
            characters_list = advanced_characters.get('characters', [])

        parent = {
            "characters": characters_list,
            "chapters": chapters
        }

        parent_path = Path("data/processed/dialogues") / f"{book_id}_parent.json"
        parent_path.parent.mkdir(parents=True, exist_ok=True)
        with open(parent_path, 'w', encoding='utf-8') as fh:
            json.dump(parent, fh, indent=2, ensure_ascii=False)

        logger.info(f"Parent dialogue JSON created at {parent_path} with {len(chapters)} chapters")
        return str(parent_path), len(chapters)
    except Exception as e:
        logger.warning(f"Failed to build parent dialogue JSON: {e}")
        raise

@app.get("/results/{book_id}")
async def get_processing_results(book_id: str):
    """Get processing results for a specific book"""
    
    book_output_dir = output_base_dir / book_id
    
    if not book_output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for book_id: {book_id}")
    
    try:
        # Get list of generated files
        generated_files = [f.name for f in book_output_dir.iterdir() if f.is_file()]
        
        # Extract statistics
        entities_count, quotes_count, characters_count = extract_statistics(book_output_dir)
        
        # Try to read the main book JSON file
        book_file = book_output_dir / f"{book_id}.book"
        book_data = None
        if book_file.exists():
            try:
                with open(book_file, 'r', encoding='utf-8') as f:
                    book_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse book JSON: {e}")
        
        return {
            "status": "success",
            "book_id": book_id,
            "output_directory": str(book_output_dir),
            "files_generated": generated_files,
            "entities_count": entities_count,
            "quotes_count": quotes_count,
            "characters_count": characters_count,
            "book_data": book_data
        }
        
    except Exception as e:
        logger.error(f"Error retrieving results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")

@app.get("/results/{book_id}/file/{filename}")
async def get_result_file(book_id: str, filename: str):
    """Get a specific result file for a book"""
    
    book_output_dir = output_base_dir / book_id
    file_path = book_output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as JSON if it's a .book file
        if filename.endswith('.book'):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass
        
        return {
            "status": "success",
            "book_id": book_id,
            "filename": filename,
            "content": content
        }
        
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

@app.delete("/results/{book_id}")
async def delete_processing_results(book_id: str):
    """Delete processing results for a specific book"""
    
    book_output_dir = output_base_dir / book_id
    
    if not book_output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for book_id: {book_id}")
    
    try:
        import shutil
        shutil.rmtree(book_output_dir)
        
        return {
            "status": "success",
            "message": f"Results for book_id '{book_id}' deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete results: {str(e)}")

@app.get("/books")
async def list_processed_books():
    """List all processed books"""
    
    try:
        processed_books = []
        
        for book_dir in output_base_dir.iterdir():
            if book_dir.is_dir():
                # Get basic info about the book
                generated_files = [f.name for f in book_dir.iterdir() if f.is_file()]
                entities_count, quotes_count, characters_count = extract_statistics(book_dir)
                
                processed_books.append({
                    "book_id": book_dir.name,
                    "output_directory": str(book_dir),
                    "files_generated": generated_files,
                    "entities_count": entities_count,
                    "quotes_count": quotes_count,
                    "characters_count": characters_count
                })
        
        return {
            "status": "success",
            "total_books": len(processed_books),
            "books": processed_books
        }
        
    except Exception as e:
        logger.error(f"Error listing books: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list books: {str(e)}")

if __name__ == "__main__":
    # Run the microservice
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,  # Different port from OpenVoice service
        reload=True,
        log_level="info"
    )