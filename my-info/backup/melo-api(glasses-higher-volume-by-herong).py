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
import os

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
    
    # Download required NLTK resources only if not already present
    import nltk
    try:
        print("Checking NLTK resources...")
        nltk_data_path = nltk.data.path[0]
        
        # Define the resources we need
        resources = [
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
        ]
        
        # Check and download only missing resources
        for resource_path, resource_name in resources:
            full_path = os.path.join(nltk_data_path, resource_path)
            if not os.path.exists(full_path):
                print(f"Downloading missing NLTK resource: {resource_name}...")
                nltk.download(resource_name)
            else:
                print(f"NLTK resource {resource_name} already present, skipping download")
        
        # Special handling for averaged_perceptron_tagger_eng if it's still missing
        eng_tagger_path = os.path.join(nltk_data_path, 'taggers', 'averaged_perceptron_tagger_eng')
        if not os.path.exists(eng_tagger_path):
            # If the eng-specific version can't be downloaded directly
            # Try to copy the regular one
            src_path = os.path.join(nltk_data_path, 'taggers', 'averaged_perceptron_tagger')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(eng_tagger_path), exist_ok=True)
            
            # Copy if source exists
            if os.path.exists(src_path):
                print(f"Copying tagger from {src_path} to {eng_tagger_path}")
                import shutil
                shutil.copytree(src_path, eng_tagger_path, dirs_exist_ok=True)
        
        print("NLTK resources check completed")
    except Exception as e:
        print(f"Error checking/downloading NLTK resources: {e}")
    
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
        
        # Check if speaker_id is an integer (direct ID) or a string (lookup key)
        if isinstance(speaker_id, int):
            # If it's already an integer, use it directly
            print(f"Using direct speaker_id integer: {speaker_id}")
            # Verify it's in range for the model
            max_id = max(model.hps.data.spk2id.values())
            if speaker_id > max_id:
                print(f"Warning: speaker_id {speaker_id} exceeds max ID {max_id}, using default")
                if language == "EN":
                    speaker_id = model.hps.data.spk2id["EN-Default"]
                    print(f"Using EN-Default speaker (American accent) for English: {speaker_id}")
                else:
                    speaker_id = list(model.hps.data.spk2id.values())[0]
        else:
            # Original string-based lookup logic
            if language == "EN" and (speaker_id is None or speaker_id == "" or speaker_id not in model.hps.data.spk2id):
                speaker_id = model.hps.data.spk2id["EN-Default"]
                print(f"Using EN-Default speaker (American accent) for English: {speaker_id}")
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
            
            # Check if audio is valid
            if len(audio) == 0 or np.isnan(audio).any():
                raise HTTPException(status_code=500, detail="Generated audio is invalid or empty")
            
            # Normalize audio to increase volume before writing to MP3
            volume_multiplier = 12.6  # Equivalent to 22dB increase for outdoor use
                
            # Log volume multiplier value
            print(f"Volume multiplier: {volume_multiplier:.1f}")
                
            normalized_audio = audio * volume_multiplier
            # Clip to avoid distortion
            normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
            
            sf.write(wav_io, normalized_audio, model.hps.data.sampling_rate, format="WAV")
            wav_io.seek(0)
            
            # Try MP3 conversion up to 3 times
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Convert to MP3 using torchaudio
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(wav_io)
                    
                    # Convert to MP3
                    mp3_io = io.BytesIO()
                    torchaudio.save(mp3_io, waveform, sample_rate, format="mp3")
                    audio_io = mp3_io
                    media_type = "audio/mpeg"
                    filename = "output.mp3"
                    break  # Success, exit the retry loop
                except RuntimeError as e:
                    print(f"MP3 conversion attempt {attempt+1}/{max_attempts} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        # Reset WAV IO for next attempt
                        wav_io.seek(0)
                    else:
                        # All attempts failed, raise an appropriate error
                        print(f"All MP3 conversion attempts failed")
                        raise HTTPException(status_code=500, detail="Failed to generate MP3 audio after multiple attempts")
            # After the loop, audio_io will contain the MP3 data if conversion succeeded
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
