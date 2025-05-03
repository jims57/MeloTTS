import torch
import numpy as np
import io
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import time
from melo.api import TTS

# API model for TTS request
class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[int] = 0
    language: Optional[str] = "ZH"
    speed: Optional[float] = 1.0
    audio_format: Optional[str] = "wav"  # wav or mp3
    sdp_ratio: Optional[float] = 0.2
    noise_scale: Optional[float] = 0.6
    noise_scale_w: Optional[float] = 0.8

# Initialize FastAPI app
app = FastAPI()

# Global variables to store models
global_models = {}

def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    device = get_device()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    
    # Initialize models for different languages
    languages = ["ZH", "EN"]
    for language in languages:
        print(f"Loading {language} model...")
        model = TTS(language=language, device=device)
        global_models[language] = model
        print(f"{language} model loaded")

@app.get("/")
async def root():
    return {"message": "MeloTTS API is running"}

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    start_time = time.time()
    print(f"API start time: {time.strftime('%H:%M:%S.%f')[:-3]}")
    
    # Get the appropriate model
    language = request.language
    if language not in global_models:
        raise HTTPException(status_code=400, detail=f"Language '{language}' not supported")
    
    model = global_models[language]
    
    try:
        # Get speaker ID
        speaker_id = request.speaker_id
        if speaker_id not in model.hps.data.spk2id:
            # Use first available speaker if specified one doesn't exist
            speaker_id = list(model.hps.data.spk2id.values())[0]
        else:
            speaker_id = model.hps.data.spk2id[speaker_id]
        
        # Generate audio
        print(f"Generating audio for text: {request.text[:50]}{'...' if len(request.text) > 50 else ''}")
        
        before_inference_time = time.time()
        elapsed_since_start = (before_inference_time - start_time) * 1000
        print(f"Time before inference: {elapsed_since_start:.2f} ms since start")
        
        audio = model.tts_to_file(
            text=request.text,
            speaker_id=speaker_id,
            output_path=None,  # Don't save to file
            sdp_ratio=request.sdp_ratio,
            noise_scale=request.noise_scale,
            noise_scale_w=request.noise_scale_w,
            speed=request.speed,
            quiet=True
        )
        
        after_inference_time = time.time()
        elapsed_since_start = (after_inference_time - start_time) * 1000
        elapsed_since_last = (after_inference_time - before_inference_time) * 1000
        print(f"Time after inference: {elapsed_since_start:.2f} ms since start, {elapsed_since_last:.2f} ms since before inference")
        
        # Create in-memory file
        audio_io = io.BytesIO()
        
        if request.audio_format.lower() == "wav":
            import soundfile as sf
            sf.write(audio_io, audio, model.hps.data.sampling_rate, format="WAV")
            media_type = "audio/wav"
            filename = "output.wav"
        elif request.audio_format.lower() == "mp3":
            import soundfile as sf
            # First write as WAV to memory
            wav_io = io.BytesIO()
            sf.write(wav_io, audio, model.hps.data.sampling_rate, format="WAV")
            wav_io.seek(0)
            
            # Convert to MP3 using torchaudio
            import torchaudio
            waveform, sample_rate = torchaudio.load(wav_io)
            
            # Convert to MP3
            mp3_io = io.BytesIO()
            torchaudio.save(mp3_io, waveform, sample_rate, format="mp3")
            audio_io = mp3_io
            media_type = "audio/mpeg"
            filename = "output.mp3"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {request.audio_format}")
        
        audio_io.seek(0)
        
        first_byte_time = time.time()
        elapsed_since_start = (first_byte_time - start_time) * 1000
        elapsed_since_last = (first_byte_time - after_inference_time) * 1000
        print(f"Time to send first byte: {elapsed_since_start:.2f} ms since start, {elapsed_since_last:.2f} ms since after inference")
        
        generation_time = time.time() - start_time
        print(f"Audio generated in {generation_time:.2f} seconds")
        
        return StreamingResponse(
            audio_io, 
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9003)
