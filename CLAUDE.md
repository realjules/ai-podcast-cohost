# AI Podcast Co-Host Development Guidelines

## Build/Run Commands
- Install dependencies: `pip install -r requirements.txt`
- Run application: `python run.py` or `python app/main.py`
- Server runs at: `http://127.0.0.1:53997`
- Mock mode fallback: Open `app/static/index.html` in a browser when dependencies aren't available

## Development Modes
- **Full Mode**: All dependencies installed and API keys configured
- **Mock Mode**: Simplified interface that works without dependencies for testing UI/UX

## Code Style Guidelines
- **Imports**: Standard library first, then third-party modules, then local imports
- **Type hints**: Use typing module for all function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Path handling**: Use pathlib.Path instead of string concatenation
- **Error handling**: Use try/except with specific exceptions, raise HTTPException for API errors
- **Async/await**: Use for all IO operations and API endpoints

## Code Organization
- `run.py`: Entry point script to run the application
- `app/main.py`: FastAPI routes and server configuration
- `app/utils.py`: Core functionality classes (AudioProcessor, DocumentProcessor, ConversationManager)
- `app/templates/index.html`: Main frontend interface
- `app/static/index.html`: Simplified mock interface for development
- Static assets stored in app/static directory
- Clear separation between UI, API endpoints, and business logic

## Dependencies
- FastAPI framework with Uvicorn ASGI server
- OpenAI for language models, audio transcription, and text-to-speech
- LangChain with FAISS for document processing
- Environment variables in .env file for API keys