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
    
    # Download required NLTK resources
    import nltk
    try:
        print("Downloading required NLTK resources...")
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        
        # Add the specific resource needed for English
        try:
            nltk.download('averaged_perceptron_tagger_eng')
        except:
            # If the eng-specific version can't be downloaded directly
            # Try to copy the regular one
            import os
            import shutil
            nltk_data_path = nltk.data.path[0]
            src_path = os.path.join(nltk_data_path, 'taggers', 'averaged_perceptron_tagger')
            dst_path = os.path.join(nltk_data_path, 'taggers', 'averaged_perceptron_tagger_eng')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copy if source exists
            if os.path.exists(src_path):
                print(f"Copying tagger from {src_path} to {dst_path}")
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
    
    # Initialize models for all supported languages
    languages = ["ZH", "EN", "ES", "FR", "JP", "KR"]
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
    
    # Get the appropriate model - convert language to uppercase
    language = request.language.upper() if request.language else "ZH"
    if language not in global_models:
        raise HTTPException(status_code=400, detail=f"Language '{language}' not supported")
    
    model = global_models[language]
    
    try:
        # Get speaker ID
        speaker_id = request.speaker_id
        if language == "EN" and (speaker_id is None or speaker_id == "" or speaker_id not in model.hps.data.spk2id):
            speaker_id = model.hps.data.spk2id["EN-Default"]
            print(f"Using EN-Default speaker for English: {speaker_id}")
        elif speaker_id not in model.hps.data.spk2id:
            # Use first available speaker if specified one doesn't exist
            speaker_id = list(model.hps.data.spk2id.values())[0]
            print(f"Using fallback speaker_id: {speaker_id}")
        else:
            speaker_id = model.hps.data.spk2id[speaker_id]
            print(f"Using requested speaker_id: {speaker_id}")
        
        print(f"Available speakers for {language}: {model.hps.data.spk2id}")
        
        # Generate audio
        print(f"Generating audio for text: {request.text[:50]}{'...' if len(request.text) > 50 else ''}")
        
        before_inference_time = time.time()
        elapsed_since_start = (before_inference_time - start_time) * 1000
        print(f"Time before inference: {elapsed_since_start:.2f} ms since start")
        
        try:
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
        except Exception as inference_error:
            print(f"Inference error: {str(inference_error)}")
            print(f"Error type: {type(inference_error)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
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
        print(f"Error in generate_tts: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9003)
