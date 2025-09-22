# BookNLP API Service

A FastAPI-based microservice for processing literary texts using BookNLP. This service accepts text input via HTTP requests instead of requiring text files, making it easy to integrate into other applications.

## Features

- **Text Input via API**: Send text directly in HTTP request body
- **RESTful API**: Clean REST endpoints for processing and retrieving results
- **Persistent Storage**: Results stored in the same location as before (`modelsbooknlp/output`)
- **Real-time Processing**: Process books on-demand with real-time status updates
- **Comprehensive Analysis**: Entity extraction, quote attribution, coreference resolution, and more

## API Endpoints

### Core Endpoints

- `GET /` - Service status and information
- `GET /health` - Health check endpoint
- `POST /process` - Process book text using BookNLP
- `GET /results/{book_id}` - Get processing results for a specific book
- `GET /results/{book_id}/file/{filename}` - Get a specific result file
- `GET /books` - List all processed books
- `DELETE /results/{book_id}` - Delete processing results for a book

### Example Usage

#### Process a Book

```bash
curl -X POST "http://localhost:8002/process" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Chapter 1: John walked into the room. \"Hello Mary,\" he said...",
       "book_id": "my_novel_001",
       "title": "My Novel",
       "author": "Author Name"
     }'
```

#### Get Results

```bash
curl "http://localhost:8002/results/my_novel_001"
```

#### List All Books

```bash
curl "http://localhost:8002/books"
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- BookNLP models (already configured in your setup)
- Required Python packages (see requirements.txt)

### Quick Start

1. **Navigate to the service directory:**
   ```bash
   cd microservices/booknlp_service
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the service:**
   ```bash
   python app.py
   ```
   
   Or use the PowerShell startup script:
   ```powershell
   .\start_service.ps1
   ```

4. **Test the service:**
   ```bash
   python test_api.py
   ```

The service will start on `http://localhost:8002`

### API Documentation

Once the service is running, you can access:
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## Configuration

The service uses the same BookNLP model configuration as your existing setup:

```python
model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "custom",
    "entity_model_path": "C:\\Users\\Adhyan\\booknlp_models\\entities_google_bert_uncased_L-6_H-768_A-12-v1.0_modified.model",
    "coref_model_path": "C:\\Users\\Adhyan\\booknlp_models\\coref_google_bert_uncased_L-12_H-768_A-12-v1.0_modified.model",
    "quote_attribution_model_path": "C:\\Users\\Adhyan\\booknlp_models\\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1_modified.model",
    "bert_model_path": "C:\\Users\\Adhyan\\.cache\\huggingface\\hub"
}
```

## Output Structure

Results are stored in: `C:/Users/Adhyan/OneDrive/Desktop/EchoTales-Enhanced/modelsbooknlp/output/{book_id}/`

Generated files include:
- `{book_id}.book` - Main JSON output with all analysis
- `{book_id}.entities` - Named entities
- `{book_id}.quotes` - Extracted quotes
- `{book_id}.tokens` - Tokenized text
- And other BookNLP output files

## Request/Response Examples

### Process Book Request
```json
{
  "text": "Your book text here...",
  "book_id": "optional_custom_id",
  "title": "Book Title",
  "author": "Author Name",
  "pipeline": "entity,quote,supersense,event,coref"
}
```

### Process Book Response
```json
{
  "status": "success",
  "message": "Book processing completed successfully",
  "book_id": "book_abc123",
  "processing_time": 45.67,
  "output_directory": "C:/Users/Adhyan/.../output/book_abc123",
  "files_generated": ["book_abc123.book", "book_abc123.entities", "book_abc123.quotes"],
  "entities_count": 25,
  "quotes_count": 12,
  "characters_count": 8
}
```

## Testing

Run the test suite to verify everything is working:

```bash
python test_api.py
```

The test script will:
- Check if the API is running
- Test all endpoints
- Process sample text
- Verify results retrieval

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `app.py` (line 351)
2. **Model files not found**: Verify model paths in the configuration
3. **BookNLP import error**: Ensure BookNLP is properly installed
4. **Timeout errors**: Large texts may take several minutes to process

### Logs

The service logs important information to the console. Look for:
- ✓ Model file found messages during startup
- Processing progress during book analysis
- Error messages for troubleshooting

## Integration

To integrate this service into other applications:

1. **Start the service**: Ensure it's running on port 8002
2. **Make HTTP requests**: Use any HTTP client to send requests
3. **Handle responses**: Process the JSON responses in your application
4. **Retrieve results**: Use the results endpoints to get processed data

## Sharing with Others

To share this service with others:

1. **Provide the service files**: Share the entire `booknlp_service` folder
2. **Ensure they have BookNLP models**: They need the same model files
3. **Share the startup instructions**: Point them to this README
4. **Configure firewall**: If sharing across networks, ensure port 8002 is accessible

The service runs on `0.0.0.0:8002`, making it accessible from other machines on the network.