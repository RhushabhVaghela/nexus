import sys
import os
import torch
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import from scripts/ and src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'src'))

from scripts.inference import load_student, load_knowledge_engine, generate_response

app = FastAPI(title="Nexus API Server")

# Allow Dashboard to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
TOKENIZER = None
KNOWLEDGE = None

def generate_video_from_response(prompt: str, response_text: str) -> str:
    """
    Generate a video based on the prompt and response.
    
    In production, this would:
    1. Use a text-to-video model (e.g., CogVideo, VideoCrafter, or similar)
    2. Or generate a Remotion composition for server-side rendering
    3. Return a URL to the generated video
    
    For now, returns None to indicate video generation is not configured.
    """
    logger.info(f"Video generation requested for prompt: {prompt[:50]}...")
    
    # Check if Remotion is available for video generation
    remotion_dir = os.path.join(BASE_DIR, "remotion")
    if os.path.exists(remotion_dir):
        logger.info("Remotion directory found. Video generation would use Remotion.")
        # In a full implementation, this would:
        # 1. Generate a Remotion composition based on the response
        # 2. Trigger a render job
        # 3. Return the eventual video URL
        return None
    
    # Check for alternative video generation capabilities
    video_gen_module = os.path.join(BASE_DIR, "src", "video_generation.py")
    if os.path.exists(video_gen_module):
        logger.info("Video generation module found.")
        return None
    
    logger.warning("No video generation capability configured. Returning None.")
    return None

def generate_tsx_preview(prompt: str, response_text: str) -> str:
    """
    Generate a TSX preview for the response.
    
    Creates a simple React component preview of the response.
    """
    escaped_response = response_text.replace('"', '\\"').replace('\n', '\\n')
    return f"""// Nexus Explainer Component
// Generated from: {prompt[:50]}...
export const ExplanationSequence = () => {{
  return (
    <div className="explanation-container">
      <h2>Generated Explanation</h2>
      <div className="content">
        {escaped_response[:200]}{"..." if len(escaped_response) > 200 else ""}
      </div>
    </div>
  );
}};"""

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.on_event("startup")
async def startup_event():
    global MODEL, TOKENIZER, KNOWLEDGE
    release_path = os.path.join(BASE_DIR, "nexus-release-v1")
    logger.info(f"[Server] Loading Nexus Model from {release_path}...")
    try:
        MODEL, TOKENIZER = load_student(release_path)
        KNOWLEDGE = load_knowledge_engine(release_path)
        logger.info("✅ Server Ready.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Server starts even if model loading fails
        # Endpoints will return appropriate errors

@app.get("/")
async def root():
    model_status = "loaded" if MODEL is not None else "not_loaded"
    return {
        "status": "online",
        "model": "Nexus-v1.0-Student",
        "model_status": model_status
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not MODEL or not TOKENIZER:
        logger.error("Model not loaded, cannot generate response")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"[API] Received Prompt: {req.prompt[:50]}...")
    try:
        # We manually call generate_response
        # Note: generate_response in scripts/inference.py prints to stdout, 
        # but also returns the full decoded string.
        response_text = generate_response(
            MODEL, 
            TOKENIZER, 
            req.prompt, 
            knowledge_tower=KNOWLEDGE,
            max_new_tokens=req.max_tokens
        )
        
        # Attempt video generation (may return None if not configured)
        video_url = generate_video_from_response(req.prompt, response_text)
        
        # Generate TSX preview
        tsx_preview = generate_tsx_preview(req.prompt, response_text)
        
        response = {
            "response": response_text,
            "tsx_preview": tsx_preview
        }
        
        # Only include video_url if video generation succeeded
        if video_url:
            response["video_url"] = video_url
        else:
            response["video_status"] = "not_configured"
            logger.info("Video generation not configured. Set up Remotion or a text-to-video model to enable.")
        
        return response
        
    except Exception as e:
        logger.error(f"[API Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
