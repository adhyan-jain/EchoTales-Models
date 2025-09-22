# Enhanced Character Processing with Gemini AI

This enhanced character processor uses Google's Gemini AI to create detailed, literary-quality character profiles and physical descriptions for your novel characters.

## Features

### 🤖 AI-Powered Enhancements
- **Character Analysis**: Deep personality insights using Gemini AI
- **Physical Descriptions**: Detailed, literary-quality appearance descriptions
- **Character Background**: Inferred relationships, backgrounds, and story arcs
- **Dialogue Style Analysis**: How characters speak and express themselves

### 📊 Advanced Processing
- Character consolidation and merging
- Sophisticated quote attribution
- Multi-method personality analysis
- Enhanced gender and age classification
- Pronoun filtering and cleanup

## Setup

### 1. Install Dependencies
```bash
pip install google-generativeai
```

### 2. Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 3. Configure Environment
Copy the existing template and configure:
```bash
# Copy the template
cp .env.template .env

# Edit .env and update the Gemini API key:
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_CHARACTER_ENHANCEMENT=True
GEMINI_ENHANCEMENT_THRESHOLD=0.3
GEMINI_RATE_LIMIT_DELAY=1
```

Or set it as an environment variable:
```bash
# Windows
set GEMINI_API_KEY=your_actual_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_actual_api_key_here
```

## Usage

### Basic Usage
```python
from advanced_character_processor import AdvancedCharacterProcessor

# Initialize with Gemini API key
processor = AdvancedCharacterProcessor(gemini_api_key="your_api_key")

# Process your book
output_path, characters = processor.process_book("your_book_id")

print(f"Characters saved to: {output_path}")
```

### Command Line Usage
```bash
python advanced_character_processor.py
```

## Output Structure

The enhanced character.json file includes:

### Basic Character Data
- `character_id`: Unique identifier
- `name`: Character name
- `gender`: Classified gender
- `age_group`: Age classification
- `mention_count`: Frequency of mentions

### AI-Enhanced Data
- `gemini_enhanced`: AI analysis including:
  - `refined_personality`: Enhanced personality insights
  - `character_background`: Likely background and history
  - `key_relationships`: Important relationships
  - `character_arc`: Role in the story
  - `dialogue_style`: Speaking patterns

### Physical Descriptions
- `physical_description`: Enhanced AI-generated description
- `physical_description_enhanced`: Detailed literary description
- `physical_description_original`: Original generated description
- Individual traits: `height`, `build`, `hair_color`, `eye_color`, etc.

### Analysis Data
- `personality`: Big Five personality scores
- `quotes`: Sample dialogue with attribution
- `narrator_mentions`: Third-person references
- `character_analysis`: Processing statistics

## Example Enhanced Description

**Before (Basic):**
> "A well-built adult with dark brown hair that is typically well-groomed. They have brown eyes and medium skin..."

**After (Gemini Enhanced):**
> "Marcus stood with the measured bearing of someone accustomed to command, his broad shoulders carrying an invisible weight of responsibility. His dark eyes, sharp with intelligence, held a warmth that contrasted with his otherwise stern demeanor. Silver streaked through his carefully maintained beard, lending him an air of distinguished authority that made even seasoned soldiers straighten in his presence..."

## Cost Management

- Only characters with importance score > 0.3 are enhanced by default
- Rate limiting prevents API quota exhaustion
- Fallback descriptions ensure all characters get descriptions

## Configuration Options

### Environment Variables (.env file)
```bash
# Character Enhancement Settings
GEMINI_API_KEY=your_api_key_here
GEMINI_CHARACTER_ENHANCEMENT=True
GEMINI_ENHANCEMENT_THRESHOLD=0.3  # Importance score threshold
GEMINI_RATE_LIMIT_DELAY=1         # Seconds between API calls
```

### Python Configuration
```python
processor = AdvancedCharacterProcessor(
    output_dir="data/processed",
    characters_dir="data/processed/characters", 
    gemini_api_key="your_key"  # Optional: override env variable
)
```

## Error Handling

The system gracefully handles:
- Missing API keys (falls back to basic processing)
- API rate limits and errors
- Network connectivity issues
- Invalid API responses

## File Structure

```
your_project/
├── advanced_character_processor.py  # Enhanced processor
├── .env.template                    # Environment template (existing)
├── .env                            # Your actual config (copy from template)
├── data/processed/characters/       # Output directory
└── README_CHARACTER_ENHANCEMENT.md # This file
```

## Tips for Best Results

1. **Set your API key properly** - Check environment variables
2. **Monitor API usage** - Gemini has usage limits
3. **Review generated content** - AI output may need editing
4. **Backup original data** - Enhanced processing modifies character data
5. **Adjust importance threshold** - Control which characters get AI enhancement

## Troubleshooting

### No API Key Found
```
⚠ No Gemini API key found in environment variables
Set GEMINI_API_KEY environment variable to enable AI enhancement
```
**Solution**: Set the GEMINI_API_KEY environment variable

### API Rate Limits
The system includes automatic rate limiting, but if you hit quotas:
- Wait for quota to reset
- Increase `rate_limit_delay` in the GeminiCharacterEnhancer class

### Poor Quality Descriptions
- Check that character has sufficient quotes and context
- Verify the character importance score is above threshold
- Review the original BookNLP analysis quality

## Support

For issues related to:
- **BookNLP processing**: Check your input text quality
- **Gemini API**: Visit [Google AI documentation](https://ai.google.dev/docs)
- **Character analysis**: Review the character consolidation settings

## Version History

- **v4.0**: Added Gemini AI integration for character enhancement
- **v3.0**: Advanced character processing with personality analysis
- **v2.0**: Quote attribution and character merging
- **v1.0**: Basic BookNLP character extraction