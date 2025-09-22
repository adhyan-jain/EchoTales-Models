#!/usr/bin/env python3
"""
EchoTales Development Server Launcher

This script launches the EchoTales development server with minimal dependencies.
It handles missing components gracefully and provides fallbacks.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_minimal_env_file():
    """Create a minimal .env file if it doesn't exist"""
    env_file = project_root / ".env"
    
    if not env_file.exists():
        logger.info("Creating minimal .env file...")
        
        minimal_env = """# Minimal EchoTales Environment Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
LOG_LEVEL=INFO
SECRET_KEY=dev-secret-key-change-in-production
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Optional API keys (leave empty for mock mode)
OPENAI_API_KEY=
GEMINI_API_KEY=
STABILITY_API_KEY=
HUGGINGFACE_API_TOKEN=

# Mock mode settings
ENABLE_MOCK_APIS=True
MOCK_RESPONSE_DELAY=1
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(minimal_env)
        
        logger.info(f"Created {env_file}")


def ensure_directories():
    """Ensure required directories exist"""
    required_dirs = [
        "data/cache",
        "data/output/images", 
        "data/output/audio",
        "data/voice_dataset",
        "logs",
        "models"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Required directories created")


def create_minimal_flask_app():
    """Create a minimal Flask app that works without external dependencies"""
    
    try:
        from flask import Flask, jsonify
        from flask_cors import CORS
        from datetime import datetime
        import uuid
        
    except ImportError as e:
        logger.error(f"Flask not installed. Please run: pip install flask flask-cors")
        logger.error(f"Or install all dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    
    app = Flask(__name__)
    CORS(app)
    
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Basic health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0.0',
            'mode': 'development',
            'components': {
                'flask': True,
                'mock_mode': os.getenv('ENABLE_MOCK_APIS', 'True').lower() == 'true'
            }
        }), 200
    
    @app.route('/api/status', methods=['GET'])
    def api_status():
        """API status endpoint"""
        return jsonify({
            'api_version': '2.0.0',
            'status': 'running',
            'endpoints': [
                '/health',
                '/api/status',
                '/api/novels/process (POST)',
                '/api/images/generate (POST)'
            ],
            'mock_mode': os.getenv('ENABLE_MOCK_APIS', 'True').lower() == 'true'
        }), 200
    
    @app.route('/api/novels/process', methods=['POST'])
    def process_novel_mock():
        """Mock novel processing endpoint"""
        try:
            from flask import request
            
            data = request.get_json() or {}
            book_id = data.get('book_id', str(uuid.uuid4()))
            title = data.get('title', 'Untitled')
            
            # Mock response
            mock_response = {
                'book_id': book_id,
                'title': title,
                'status': 'processed',
                'mock_mode': True,
                'characters': [
                    {
                        'id': str(uuid.uuid4()),
                        'name': 'Sample Character',
                        'personality': {
                            'traits': ['brave', 'intelligent'],
                            'emotional_range': 0.7
                        }
                    }
                ],
                'scenes': [
                    {
                        'id': str(uuid.uuid4()),
                        'chapter': 1,
                        'setting': 'Castle courtyard',
                        'characters': ['Sample Character']
                    }
                ],
                'processing_time': '2.3s',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return jsonify(mock_response), 200
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    @app.route('/api/images/generate', methods=['POST'])
    def generate_image_mock():
        """Mock image generation endpoint"""
        try:
            from flask import request
            
            data = request.get_json() or {}
            
            mock_response = {
                'job_id': str(uuid.uuid4()),
                'status': 'completed',
                'mock_mode': True,
                'image_url': 'https://via.placeholder.com/512x512/FF6B6B/FFFFFF?text=Sample+Character',
                'character': data.get('character_name', 'Unknown Character'),
                'prompt': data.get('prompt', 'A character portrait'),
                'generation_time': '3.1s',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return jsonify(mock_response), 200
            
        except Exception as e:
            return jsonify({'error': f'Image generation failed: {str(e)}'}), 500
    
    return app


def main():
    """Main launcher function"""
    logger.info("Starting EchoTales Development Server...")
    
    try:
        # Create minimal setup
        create_minimal_env_file()
        ensure_directories()
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")
        
    except ImportError:
        logger.warning("python-dotenv not available, continuing without .env loading")
    
    # Create and run Flask app
    app = create_minimal_flask_app()
    
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Server starting at http://{host}:{port}")
    logger.info("Available endpoints:")
    logger.info("  - GET  /health - Health check")
    logger.info("  - GET  /api/status - API status")
    logger.info("  - POST /api/novels/process - Process novel (mock)")
    logger.info("  - POST /api/images/generate - Generate image (mock)")
    logger.info("\nPress Ctrl+C to stop the server")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()