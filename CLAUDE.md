# AI Podcast Co-Host Development Guidelines

## Build/Run Commands
- Install dependencies: `pip install -r requirements.txt`
- Run application: `python app/main.py`
- Server runs at: `http://localhost:53269`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party modules, then local imports
- **Type hints**: Use typing module for all function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Path handling**: Use pathlib.Path instead of string concatenation
- **Error handling**: Use try/except with specific exceptions, raise HTTPException for API errors
- **Async/await**: Use for all IO operations and API endpoints

## Code Organization
- `app/main.py`: FastAPI routes and server configuration
- `app/utils.py`: Core functionality classes
- `app/templates/index.html`: Frontend interface
- Static assets stored in app/static directory
- Clear separation between UI, API endpoints, and business logic

## Dependencies
- FastAPI framework with Uvicorn ASGI server
- OpenAI for language models and audio transcription
- ElevenLabs for text-to-speech
- LangChain with FAISS for document processing