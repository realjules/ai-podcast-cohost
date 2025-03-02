# AI Podcast Co-Host

An interactive AI-powered podcast co-host that can engage in natural conversations, process context from documents, and respond with voice.

## Features

- 🎙️ **Voice Interaction**
  - Real-time voice recording
  - Speech-to-text using OpenAI's Whisper API
  - Text-to-speech using OpenAI's TTS API

- 📚 **Document Processing**
  - PDF upload and processing
  - Vector storage using FAISS
  - Context-aware responses

- 🤖 **AI Conversation**
  - GPT-4 powered responses
  - Memory of conversation history
  - Natural, podcast-style interaction

- 💻 **Modern Interface**
  - Clean UI with Tailwind CSS
  - Real-time audio visualization
  - Easy-to-use controls

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, JavaScript, Tailwind CSS
- **AI/ML**: 
  - OpenAI GPT-4, Whisper & TTS
  - LangChain
  - FAISS Vector Store

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-podcast-cohost.git
   cd ai-podcast-cohost
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Run the application:
   ```bash
   python app/main.py
   ```

5. Open your browser and navigate to `http://localhost:53997`

## Usage

1. **Upload Context (Optional)**
   - Click "Upload PDF" to add background knowledge
   - The system will process and index the content

2. **Start Conversation**
   - Use the voice recorder: Click "Start Recording" to speak
   - Or type messages in the text input

3. **Interact with AI**
   - View your transcribed speech
   - See AI responses in text form
   - Listen to AI responses through the audio player

## Project Structure

```
ai-podcast-cohost/
├── app/
│   ├── static/
│   │   ├── uploads/    # Temporary file storage
│   │   └── audio/      # Generated audio files
│   ├── templates/
│   │   └── index.html  # Main UI template
│   ├── main.py         # FastAPI application
│   └── utils.py        # Core functionality
├── requirements.txt
└── .env
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.