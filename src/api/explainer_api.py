import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.inference.remotion_engine import RemotionExplainerEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nexus Universal Explainer API")

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance (lazy loaded)
engine = None

class ExplainerRequest(BaseModel):
    prompt: str
    model_path: str = "/mnt/e/data/output/trained/remotion-explainer"
    narrate: bool = False

class ExplainerResponse(BaseModel):
    success: bool
    video_url: str
    tsx_preview: str

@app.on_event("startup")
async def startup_event():
    global engine
    # Initialize engine with a default model path if it exists
    default_model = "/mnt/e/data/output/trained/remotion-explainer"
    if os.path.exists(default_model):
        try:
            engine = RemotionExplainerEngine(model_path=default_model)
            logger.info("Explainer Engine initialized on startup.")
        except Exception as e:
            logger.warning(f"Failed to initialize engine on startup: {e}")

@app.post("/generate", response_model=ExplainerResponse)
async def generate_explanation(request: ExplainerRequest):
    global engine
    
    if engine is None:
        try:
            engine = RemotionExplainerEngine(model_path=request.model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

    try:
        # Generate video
        video_path = engine.generate_video(
            prompt=request.prompt,
            narrate=request.narrate
        )
        
        # Read the TSX for preview
        tsx_path = engine.remotion_dir / "src" / "GeneratedScene.tsx"
        with open(tsx_path, "r") as f:
            tsx_code = f.read()
            
        return ExplainerResponse(
            success=True,
            video_url=video_path,
            tsx_preview=tsx_code
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_ready": engine is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
