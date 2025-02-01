import os
from pathlib import Path
import aiofiles
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from .utils import AudioProcessor, ConversationManager, DocumentProcessor

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
UPLOAD_DIR = Path("app/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("app/static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize managers
conversation_manager = ConversationManager()
audio_processor = AudioProcessor()

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
async def chat(text: str):
    try:
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=53269, reload=True)