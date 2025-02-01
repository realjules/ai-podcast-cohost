import os
from typing import List, Optional
import tempfile
from pathlib import Path
import openai
from elevenlabs import generate, save, set_api_key
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Configure API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    async def process_pdf(self, file_path: str) -> None:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splits = self.text_splitter.split_documents(pages)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        if not self.vector_store:
            return ""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in docs)

class AudioProcessor:
    @staticmethod
    async def transcribe_audio(audio_file_path: str) -> str:
        with open(audio_file_path, "rb") as audio_file:
            transcript = await openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text

    @staticmethod
    async def generate_speech(text: str) -> str:
        audio = generate(
            text=text,
            voice="Adam",
            model="eleven_monolingual_v1"
        )
        
        # Save to temporary file
        temp_dir = Path("app/static/audio")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / f"{hash(text)}.mp3"
        save(audio, str(temp_file))
        
        return f"/static/audio/{temp_file.name}"

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
        messages = self._build_prompt(query)
        
        response = await openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )

        ai_message = response.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": ai_message})
        
        return ai_message