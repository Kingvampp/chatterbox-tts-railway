import os
import torch
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTTS
import io
import torchaudio
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatterbox TTS API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL = None
VOICES = ["neutral", "male", "female", "young_male", "young_female", "elderly_male", "elderly_female"]
EMOTIONS = ["professional", "excited", "happy", "calm", "sad", "angry", "surprised", "confused", "friendly", "neutral"]

class TTSRequest(BaseModel):
    text: str
    voice: str = "neutral"  # Keep for API compatibility but won't be used directly
    emotion: str = "professional"  # Keep for API compatibility but won't be used directly
    exaggeration: float = 0.5
    num_inference_steps: int = 400  # Keep for API compatibility but won't be used directly

@app.on_event("startup")
async def startup_event():
    global MODEL
    logger.info("Loading Chatterbox TTS model...")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Patch torch.load to always map to CPU when needed
    original_torch_load = torch.load
    def patched_torch_load(f, *args, **kwargs):
        if not torch.cuda.is_available():
            kwargs['map_location'] = 'cpu'
        return original_torch_load(f, *args, **kwargs)
    
    # Apply the patch
    torch.load = patched_torch_load
    
    try:
        # Load the model
        MODEL = ChatterboxTTS.from_pretrained(device=device)
        
        # Use half precision if on GPU
        if device == "cuda":
            MODEL = MODEL.half()
            
        logger.info(f"✅ Chatterbox TTS loaded with {device.upper()}")
    finally:
        # Restore original torch.load
        torch.load = original_torch_load

@app.get("/health")
async def health_check():
    """Check if the server is running and model is loaded."""
    return {"status": "healthy", "model_loaded": MODEL is not None}

@app.get("/voices")
async def list_voices():
    """List available voices."""
    return {"voices": VOICES, "emotions": EMOTIONS}

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    """OpenAI-compatible endpoint for generating speech."""
    global MODEL
    
    logger.info(f"Generating speech: '{request.text[:50]}...' with voice: {request.voice}, emotion: {request.emotion}, exaggeration: {request.exaggeration}")
    
    # Calculate cfg_weight (1 - exaggeration)
    cfg_weight = 1.0 - request.exaggeration
    logger.info(f"Using cfg_weight: {cfg_weight}, exaggeration: {request.exaggeration}, voice: {request.voice}")
    
    try:
        # Generate audio - removing voice, emotion, and num_inference_steps parameters that cause errors
        audio = MODEL.generate(
            text=request.text,
            cfg_weight=cfg_weight
        )
        
        # Convert to WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, MODEL.sr, format="wav")
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        return Response(content=f"Error generating speech: {str(e)}", status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8881))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
