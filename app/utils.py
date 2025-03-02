import os
from typing import List, Optional
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import time
import random

# Import optional dependencies
try:
    import openai
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import PyPDFLoader
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    MOCK_MODE = False
except ImportError:
    print("Running in MOCK mode - some dependencies not available")
    MOCK_MODE = True

# Load environment variables
load_dotenv()

# Configure API key only if not in mock mode
if not MOCK_MODE:
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        print(f"Error setting API key: {str(e)}")
        MOCK_MODE = True

class DocumentProcessor:
    def __init__(self):
        self.vector_store = None
        if not MOCK_MODE:
            self.embeddings = OpenAIEmbeddings()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

    async def process_pdf(self, file_path: str) -> None:
        if MOCK_MODE:
            print(f"MOCK: Processing PDF {file_path}")
            return
            
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            splits = self.text_splitter.split_documents(pages)
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        if MOCK_MODE:
            return f"MOCK: Relevant context for query: {query}"
            
        if not self.vector_store:
            return ""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            print(f"Error getting relevant context: {str(e)}")
            return ""

class AudioProcessor:
    @staticmethod
    async def transcribe_audio(audio_file_path: str) -> str:
        if MOCK_MODE:
            print(f"MOCK: Transcribing audio from {audio_file_path}")
            # Return a mock transcription for testing
            mock_responses = [
                "Hi there, I'm testing this podcast co-host application.",
                "How does artificial intelligence work in podcasting?",
                "Can you tell me more about the features of this app?",
                "What do you think about current tech trends?"
            ]
            return random.choice(mock_responses)
            
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = await openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return transcript.text
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return "Audio transcription failed."

    @staticmethod
    async def generate_speech(text: str) -> str:
        if MOCK_MODE:
            print(f"MOCK: Generating speech for text: {text}")
            # In mock mode, we just return a fixed path to a default audio file
            return "/static/audio/no_audio.mp3"
            
        try:
            # Save to temporary file
            temp_dir = Path("app/static/audio")
            temp_dir.mkdir(exist_ok=True)
            
            temp_file = temp_dir / f"{hash(text)}.mp3"
            
            # Generate speech using OpenAI's TTS API
            response = await openai.audio.speech.create(
                model="tts-1",
                voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text
            )
            
            # Save the audio file
            response.stream_to_file(str(temp_file))
            
            return f"/static/audio/{temp_file.name}"
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return "/static/audio/no_audio.mp3"

class ConversationManager:
    def __init__(self):
        self.conversation_history = []
        self.doc_processor = DocumentProcessor()

    def _build_prompt(self, query: str) -> str:
        context = self.doc_processor.get_relevant_context(query)
        
        system_prompt = """You are an engaging podcast co-host. Your responses should be:
        1. Conversational and natural
        2. Informative but accessible
        3. Engaging and sometimes humorous
        4. Related to the context when available
        
        If context is provided, use it to inform your responses but don't quote it directly.
        Keep responses concise (2-3 sentences) unless specifically asked for more detail."""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})

        for msg in self.conversation_history[-4:]:  # Keep last 4 messages for context
            messages.append(msg)

        messages.append({"role": "user", "content": query})
        return messages

    async def get_response(self, query: str) -> str:
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        if MOCK_MODE:
            print(f"MOCK: Getting AI response for query: {query}")
            # Generate a mock response
            mock_responses = [
                f"That's an interesting point about {query.split()[0] if query else 'that'}. I think podcast audiences would find this engaging.",
                "I agree with your perspective. This would make for a great discussion topic on air.",
                "Let me add to that thought. The key thing to remember is how this relates to our listeners' experiences.",
                "I'd approach this from a different angle. Have you considered the implications for content creators?",
                "That reminds me of a recent study on this topic. The findings were quite surprising."
            ]
            time.sleep(0.5)  # Simulate processing time
            ai_message = random.choice(mock_responses)
            self.conversation_history.append({"role": "assistant", "content": ai_message})
            return ai_message
        
        try:
            messages = self._build_prompt(query)
            
            response = await openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )

            ai_message = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_message})
            
            return ai_message
        except Exception as e:
            print(f"Error getting AI response: {str(e)}")
            error_message = "I'm sorry, I couldn't process that request."
            self.conversation_history.append({"role": "assistant", "content": error_message})
            return error_message