import sys
import os
import torch
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.on_event("startup")
async def startup_event():
    global MODEL, TOKENIZER, KNOWLEDGE
    release_path = os.path.join(BASE_DIR, "nexus-release-v1")
    print(f"[Server] Loading Nexus Model from {release_path}...")
    MODEL, TOKENIZER = load_student(release_path)
    KNOWLEDGE = load_knowledge_engine(release_path)
    print("✅ Server Ready.")

@app.get("/")
async def root():
    return {"status": "online", "model": "Nexus-v1.0-Student"}

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not MODEL or not TOKENIZER:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    print(f"[API] Received Prompt: {req.prompt[:50]}...")
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
        
        # Dashboard expects video_url and tsx_preview (specific to explainer use case)
        # We'll provide a mock TSX based on the response
        return {
            "response": response_text,
            "video_url": "https://example.com/mock_render.mp4",
            "tsx_preview": f"// Nexus Explained: {req.prompt[:30]}\nexport const Sequence = () => {{\n  return <p>{response_text}</p>;\n}};"
        }
    except Exception as e:
        print(f"[API Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
