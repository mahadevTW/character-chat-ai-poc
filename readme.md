# Character Chat Application

A FastAPI-based web application that allows users to chat with movie characters using RAG (Retrieval-Augmented Generation) technology. The app uses movie scripts to provide contextually relevant responses in the character's voice.

## Features

- **Character Chat**: Chat with any character from supported movies
- **RAG Integration**: Uses movie scripts to provide contextually relevant responses
- **Session Management**: Maintains conversation history for each chat session
- **FastAPI Backend**: Modern, fast web framework with automatic API documentation
- **OpenAI Integration**: Powered by GPT-4 for intelligent character responses

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Movie script files in DOCX format

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd /path/to/character-chat
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   
   Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Project Structure

```
character-chat/
├── main.py              # FastAPI application entry point
├── rag_util.py          # RAG utilities for script processing
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── data/                # Movie script data
│   └── schindlers/      # Example movie data
│       └── script.docx  # Movie script file
└── readme.md           # This file
```

## Running the Application

### Development Mode (with auto-reload)
```bash
PYTHONPATH=. python3 -m uvicorn main:app --reload
```

### Production Mode
```bash
PYTHONPATH=. python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The application will be available at:
- **Local**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## API Usage

### Chat Endpoint

**POST** `/chat`

Send a message to a movie character and receive a response in character.

#### Request Body
```json
{
  "movie": "schindlers",
  "character": "Oskar Schindler",
  "message": "What motivates you to help the Jews?",
  "session_id": "optional-session-id"
}
```

#### Response
```json
{
  "reply": "Character's response in their voice...",
  "session_id": "generated-or-provided-session-id"
}
```

#### Parameters
- `movie` (string, required): Name of the movie (must match a folder in `data/`)
- `character` (string, required): Character name from the movie
- `message` (string, required): Your message to the character
- `session_id` (string, optional): Session ID to maintain conversation history

## Adding New Movies

1. **Create a movie directory** in the `data/` folder:
   ```bash
   mkdir data/your_movie_name
   ```

2. **Add the script file**:
   - Place the movie script as `script.docx` in the movie directory
   - The script should be in DOCX format
   - Example: `data/your_movie_name/script.docx`

3. **Use the movie** in your API calls:
   ```json
   {
     "movie": "your_movie_name",
     "character": "Character Name",
     "message": "Hello!"
   }
   ```

## How It Works

1. **Script Processing**: The app reads movie scripts from DOCX files
2. **Scene Segmentation**: Scripts are split into individual scenes
3. **RAG Retrieval**: When a user sends a message, the system finds the most relevant scenes using TF-IDF and cosine similarity
4. **Context Generation**: Relevant scenes are included in the system prompt
5. **Character Response**: GPT-4 generates a response in the character's voice based on the movie context

## Dependencies

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **OpenAI**: GPT-4 integration for character responses
- **python-docx**: Reading DOCX files
- **scikit-learn**: TF-IDF vectorization and similarity calculations
- **pydantic**: Data validation
- **python-dotenv**: Environment variable management

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   - Ensure you're running with `PYTHONPATH=.`
   - Check that all dependencies are installed: `pip install -r requirements.txt`

2. **OpenAI API errors**:
   - Verify your API key is correct in the `.env` file
   - Ensure you have sufficient credits in your OpenAI account

3. **Script not found errors**:
   - Check that the movie folder exists in `data/`
   - Ensure the script file is named `script.docx`
   - Verify the script file is not corrupted

4. **Port already in use**:
   - Change the port: `uvicorn main:app --port 8001`
   - Or kill the existing process using the port

### Environment Setup Issues

If you encounter permission issues on macOS/Linux:
```bash
chmod +x venv/bin/activate
```

## Development

### Adding New Features

1. **New RAG Methods**: Extend `rag_util.py` with additional retrieval methods
2. **Character Profiles**: Add character-specific prompts or behaviors
3. **Multi-language Support**: Implement language detection and translation
4. **Voice Integration**: Add text-to-speech for character responses

### Testing

Test the API using curl:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "movie": "schindlers",
       "character": "Oskar Schindler",
       "message": "Hello, how are you?"
     }'
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]