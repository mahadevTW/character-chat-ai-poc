# main.py
import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from rag_util import generate_prompt_with_rag
import requests
import re
import time

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()
templates = Jinja2Templates(directory="templates")
session_store = {}
# Cache for system prompts to avoid regenerating for same movie-character combinations
system_prompt_cache = {}


API_KEY = os.getenv("HEYGEN_API_KEY")
HEADERS = {
    "Accept": "application/json",
    "X-Api-Key": API_KEY,
    "Content-Type": "application/json"
}

# Replace with your avatar/voice IDs
DEFAULT_AVATAR_ID = "95cbb6381a7948e6a0c7e7294ab222af"
DEFAULT_VOICE_ID = "73c0b6a2e29d4d38aca41454bf58c955"


# 1. Generate avatar video
def generate_video(avatar_id, voice_id, text, width=1280, height=720):
    # Heuristic: Talking Photo IDs are 32 hex chars
    is_talking_photo = bool(re.fullmatch(r"[0-9a-f]{32}", avatar_id))

    if is_talking_photo:
        character = {
            "type": "talking_photo",
            "talking_photo_id": avatar_id,
            # Optional niceties for Avatar IV:
            "talking_style": "expressive",   # or "stable"
            "expression": "happy",           # or "default"
            "super_resolution": True
        }
    else:
        character = {
            "type": "avatar",
            "avatar_id": avatar_id,
            "avatar_style": "normal"
        }

    payload = {
        "video_inputs": [{
            "character": character,
            "voice": {
                "type": "text",
                "voice_id": voice_id,
                "input_text": text
            },
            "background": {"type": "color", "value": "#FFFFFF"}
        }],
        "dimension": {"width": width, "height": height}
    }

    resp = requests.post("https://api.heygen.com/v2/video/generate", headers=HEADERS, json=payload)
    resp.raise_for_status()
    vid = resp.json()["data"]["video_id"]
    print(f"video_id = {vid}")
    return vid


# 2. Poll until video is ready, then return video URL
def poll_video(video_id):
    # Official status endpoint
    url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    while True:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()["data"]
        print("Status:", data["status"])
        if data["status"] == "completed":
            print("Video URL:", data["video_url"])
            return data["video_url"]
        if data["status"] == "failed":
            raise RuntimeError(data.get("error"))
        time.sleep(5)
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """
    Serves the main chat interface HTML page.
    
    This endpoint renders the index.html template which contains the web interface
    for the character chat application. Users can select movies, characters, and
    start conversations through this interface.
    
    Args:
        request (Request): FastAPI request object for template rendering
        
    Returns:
        HTMLResponse: Rendered HTML page with the chat interface
    """
    return templates.TemplateResponse("index.html", {"request": request})

class ChatRequest(BaseModel):
    """
    Pydantic model for validating chat request data.
    
    This model defines the structure and validation rules for incoming chat requests.
    It ensures that all required fields are present and properly typed.
    
    Attributes:
        movie (str): The name of the movie to get character context from
        character (str): The character name to roleplay as
        message (str): The user's message to send to the character
        session_id (str, optional): Unique identifier for maintaining conversation context.
                                   If not provided, a new session will be created.
    """
    movie: str
    character: str
    message: str
    session_id: str = None

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Handles character chat requests using OpenAI GPT-4 with RAG-enhanced context.
    
    This endpoint implements a conversational AI system that:
    1. Maintains conversation history per session
    2. Generates context-aware system prompts using RAG
    3. Caches system prompts to optimize performance
    4. Sends requests to OpenAI GPT-4 for character responses
    5. Returns character responses while maintaining session state
    
    The system uses multiple caching layers:
    - Script loading is cached per movie (in rag_util.py)
    - Scene splitting is cached per movie (in rag_util.py)
    - System prompts are cached per movie-character combination
    
    Args:
        request (ChatRequest): Validated request containing movie, character, message, and optional session_id
        
    Returns:
        dict: Response containing:
              - reply (str): The character's response
              - session_id (str): Session identifier for maintaining conversation context
              
    Technical Process:
        1. Generates or retrieves session_id for conversation tracking
        2. Retrieves existing conversation history or creates new session
        3. For new sessions: generates RAG-enhanced system prompt (with caching)
        4. Adds user message to conversation history
        5. Sends complete conversation to OpenAI GPT-4
        6. Adds AI response to history and updates session store
        7. Returns response and session_id to client
        
    Note:
        - Sessions persist in memory (session_store) for the application lifecycle
        - System prompts are cached to avoid regenerating for same movie-character pairs
        - Each session maintains full conversation history for context continuity
    """
    session_id = request.session_id or str(uuid.uuid4())
    history = session_store.get(session_id, [])

    if not history:
        # Create cache key for movie-character combination
        cache_key = f"{request.movie}_{request.character}"
        
        # Check if system prompt is already cached
        if cache_key in system_prompt_cache:
            system_prompt = system_prompt_cache[cache_key]
        else:
            system_prompt = generate_prompt_with_rag(
                movie=request.movie,
                character=request.character,
                user_message=request.message
            )
            # Cache the system prompt
            system_prompt_cache[cache_key] = system_prompt
        
        history = [{"role": "system", "content": system_prompt}]

    history.append({"role": "user", "content": request.message})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=history
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    session_store[session_id] = history

    # 🔥 Generate video with HeyGen
    video_id = generate_video(DEFAULT_AVATAR_ID, DEFAULT_VOICE_ID, reply)
    video_url = poll_video(video_id)

    return {"reply": reply, "session_id": session_id, "video_url": video_url}
