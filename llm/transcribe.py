import os
import httpx
from dotenv import load_dotenv

load_dotenv()

class TranscriptionError(Exception):
    pass

def transcribe_audio(file_bytes: bytes, filename: str, timeout: float = 30.0) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise TranscriptionError("GROQ_API_KEY is missing from environment variables.")

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    files = {
        "file": (filename, file_bytes)
    }
    data = {
        "model": "whisper-large-v3",
        "response_format": "json"
    }
    
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            transcript = result.get("text", "").strip()
            
            if not transcript:
                raise TranscriptionError("Empty transcript result returned from API.")
            return transcript
            
    except httpx.HTTPStatusError as e:
        raise TranscriptionError(f"HTTP error {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        raise TranscriptionError(f"Request error: {str(e)}")
