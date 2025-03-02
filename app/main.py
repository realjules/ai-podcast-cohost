import os
from pathlib import Path
import sys

# Set MOCK_MODE flag for testing without dependencies
MOCK_MODE = False

# Try to import dependencies
try:
    import aiofiles
    from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError as e:
    print(f"WARNING: Missing dependency: {e}")
    print("Starting in simple HTTP server mode...")
    MOCK_MODE = True
    
# Continue with local imports
if not MOCK_MODE:
    try:
        from utils import AudioProcessor, ConversationManager, DocumentProcessor
    except ImportError as e:
        print(f"ERROR: Failed to import local modules: {e}")
        MOCK_MODE = True

# Create required directories
UPLOAD_DIR = Path("app/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("app/static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Only initialize FastAPI if not in mock mode
if not MOCK_MODE:
    app = FastAPI()
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files and templates
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    templates = Jinja2Templates(directory="app/templates")
    
    # Initialize managers
    conversation_manager = ConversationManager()
    audio_processor = AudioProcessor()
else:
    # Define a placeholder when in mock mode
    app = None

# Define API routes only if not in mock mode
if not MOCK_MODE:
    @app.get("/")
    async def home(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})
    
    @app.post("/upload-pdf")
    async def upload_pdf(file: UploadFile = File(...)):
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        file_path = UPLOAD_DIR / file.filename
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        await conversation_manager.doc_processor.process_pdf(str(file_path))
        return {"message": "PDF processed successfully"}
    
    @app.post("/upload-audio")
    async def upload_audio(file: UploadFile = File(...)):
        if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Save the uploaded audio file
        file_path = UPLOAD_DIR / file.filename
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Transcribe the audio
        try:
            transcription = await audio_processor.transcribe_audio(str(file_path))
            
            # Get AI response
            ai_response = await conversation_manager.get_response(transcription)
            
            # Generate speech from AI response
            audio_url = await audio_processor.generate_speech(ai_response)
            
            return {
                "transcription": transcription,
                "response": ai_response,
                "audio_url": audio_url
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up the uploaded file
            os.unlink(file_path)
    
    @app.post("/chat")
    async def chat(request: Request):
        try:
            data = await request.json()
            text = data.get("text", "")
            if not text:
                raise HTTPException(status_code=400, detail="Text is required")
                
            # Get AI response
            ai_response = await conversation_manager.get_response(text)
            
            # Generate speech from AI response
            audio_url = await audio_processor.generate_speech(ai_response)
            
            return {
                "response": ai_response,
                "audio_url": audio_url
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/conversation-history")
    async def get_conversation_history():
        return {"history": conversation_manager.conversation_history}

def run():
    """Main function to run the application"""
    if MOCK_MODE:
        print("""
=======================================================================
MOCK MODE: Started due to missing dependencies
-----------------------------------------------------------------------
To run the full application, install all dependencies:
    pip install -r requirements.txt

For now, a simplified version is available at:
    app/static/index.html

This version doesn't require external dependencies but has limited
functionality - it can't process audio or PDFs but does simulate 
basic conversation.
=======================================================================
""")
        # Create a text file that explains how to run the app properly
        with open("README.mock.txt", "w") as f:
            f.write("""
AI PODCAST CO-HOST (MOCK MODE)
==============================

The application is running in MOCK MODE due to missing dependencies.

To run the full application with all features:
1. Install all dependencies:
   pip install -r requirements.txt

2. Make sure you have valid API keys in .env file:
   OPENAI_API_KEY=your_key_here
   ELEVENLABS_API_KEY=your_key_here

3. Run the application:
   python run.py

For now, you can view a simplified version by opening:
   app/static/index.html
   
This simplified version simulates conversation but doesn't have audio
processing or document understanding capabilities.
""")
        print("Created README.mock.txt with instructions")
        
        # Exit gracefully
        print("Exiting - Please install required dependencies to use the full application")
    else:
        # Normal FastAPI startup - using the current module
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=53269, reload=False)

if __name__ == "__main__":
    run()