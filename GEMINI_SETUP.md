# Gemini API Setup

## How to set up Gemini API for gender classification

1. **Get your API key:**
   - Go to https://makersuite.google.com/app/apikey
   - Sign in with your Google account
   - Create a new API key

2. **Add your API key to the .env file:**
   - Open the `.env` file in your project root
   - Replace `your_gemini_api_key_here` with your actual API key
   - Example: `GEMINI_API_KEY=AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

3. **The system will automatically:**
   - Load your API key from the .env file
   - Use Gemini AI to classify character names as male/female
   - Fall back to other methods if the API is unavailable

## Benefits of using Gemini AI:
- More accurate gender classification based on name phonetics
- Handles international names better
- Learns from context and name patterns
- Reduces false classifications

## Fallback behavior:
If the API key is not set or the API call fails, the system will use:
- Context analysis (pronouns in text)
- Title analysis (Mr./Ms. patterns)
- Basic pattern matching

The system is designed to work even without the API key, but with reduced accuracy.
