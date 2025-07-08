import torch
import numpy as np
import io
import json
import base64
import argparse
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import time
from melo.api import TTS
import os
import asyncio

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

@app.websocket("/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    print(f"[WS-TTS] WebSocket connection established")
    
    try:
        while True:
            # Receive JSON data from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Start timing
            start_time = time.time()
            print(f"[WS-TTS] Request start time: {time.strftime('%H:%M:%S.%f')[:-3]}")
            
            # Extract parameters matching cosy-api format
            text = request_data.get("text", "")
            speaker_id = request_data.get("speakerId", 0)  # Note: speakerId not speaker_id
            save_audio_files = request_data.get("saveAudioFiles", False)
            output_sample_rate = request_data.get("outputSampleRate", 22050)
            audio_format = request_data.get("audioFormat", "mp3")  # pcm or mp3
            
            if not text:
                await websocket.send_text(json.dumps({"error": "Text is required"}))
                continue
            
            # Get the appropriate model - convert language to uppercase
            language = request_data.get("language", "ZH").upper()
            if language not in global_models:
                await websocket.send_text(json.dumps({"error": f"Language '{language}' not supported"}))
                continue
            
            model = global_models[language]
            
            try:
                # Get speaker ID
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
                
                # Setup folder for saving audio chunks if requested
                chunk_save_folder = None
                chunk_file_counter = 0
                if save_audio_files:
                    chunk_save_folder = os.path.join(os.path.dirname(__file__), "melo_audio_chunks")
                    os.makedirs(chunk_save_folder, exist_ok=True)
                    print(f"[WS-TTS] Created/verified chunk save folder: {chunk_save_folder}")
                
                # Generate audio
                print(f"Generating audio for text: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                before_inference_time = time.time()
                elapsed_since_start = (before_inference_time - start_time) * 1000
                print(f"Time before inference: {elapsed_since_start:.2f} ms since start")
                
                try:
                    # Generate full audio first
                    audio = model.tts_to_file(
                        text=text,
                        speaker_id=speaker_id,
                        output_path=None,  # Don't save to file
                        sdp_ratio=request_data.get("sdp_ratio", 0.2),
                        noise_scale=request_data.get("noise_scale", 0.6),
                        noise_scale_w=request_data.get("noise_scale_w", 0.8),
                        speed=request_data.get("speed", 1.0),
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
                
                # Resample audio if needed
                if model.hps.data.sampling_rate != output_sample_rate:
                    import torchaudio
                    resample_start = time.time()
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, 
                        model.hps.data.sampling_rate, 
                        output_sample_rate
                    )
                    audio = audio_tensor.squeeze(0).numpy()
                    resample_time = (time.time() - resample_start) * 1000
                    print(f"[WS-TTS] Resampled audio: {resample_time:.2f}ms ({model.hps.data.sampling_rate} → {output_sample_rate} Hz)")
                else:
                    print(f"[WS-TTS] No resampling needed (optimal)")
                
                # Stream audio in chunks
                chunk_duration = 1.0  # 1 second chunks
                samples_per_chunk = int(output_sample_rate * chunk_duration)
                chunk_counter = 0
                
                print(f"[WS-TTS] Streaming audio format: {audio_format}")
                print(f"[WS-TTS] Audio length: {len(audio)} samples, chunk size: {samples_per_chunk} samples")
                
                # Stream audio data in chunks
                for i in range(0, len(audio), samples_per_chunk):
                    chunk_start_time = time.time()
                    
                    # Extract audio chunk
                    audio_chunk = audio[i:i + samples_per_chunk]
                    
                    if audio_format.lower() == "pcm":
                        # Convert to 16-bit PCM
                        audio_np = audio_chunk
                        
                        # Apply volume multiplier (same logic as MP3)
                        volume_multiplier = 2.0
                        # Apply higher volume for Chinese language PCM
                        if language == "ZH":
                            volume_multiplier = 8.47  # Equivalent to 13dB increase
                            
                        # Log volume multiplier value
                        print(f"Volume multiplier: {volume_multiplier:.1f}")
                            
                        normalized_audio = audio_np * volume_multiplier
                        # Clip to avoid distortion
                        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                        
                        # Normalize to [-1, 1] range if needed
                        if normalized_audio.max() > 1.0 or normalized_audio.min() < -1.0:
                            normalized_audio = normalized_audio / max(abs(normalized_audio.max()), abs(normalized_audio.min()))
                        # Convert to 16-bit PCM
                        pcm_data = (normalized_audio * 32767).astype(np.int16).tobytes()
                        
                        # Send PCM chunk
                        await websocket.send_bytes(pcm_data)
                        
                        # Save PCM chunk if requested
                        if save_audio_files and chunk_save_folder:
                            chunk_filename = f"chunk_{chunk_counter}.pcm"
                            chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                            try:
                                with open(chunk_filepath, 'wb') as f:
                                    f.write(pcm_data)
                                print(f"[WS-TTS] Saved {chunk_filename} ({len(pcm_data)} bytes)")
                            except Exception as save_error:
                                print(f"[WS-TTS] Error saving chunk file: {save_error}")
                        
                        chunk_processing_time = (time.time() - chunk_start_time) * 1000
                        print(f"[WS-TTS] PCM chunk {chunk_counter} sent: {len(pcm_data)} bytes, time: {chunk_processing_time:.2f}ms")
                        
                    elif audio_format.lower() == "mp3":
                        # Convert chunk to MP3 using enhanced method from cosy-api.py
                        import soundfile as sf
                        # First write as WAV to memory
                        wav_io = io.BytesIO()
                        
                        # Check if audio is valid
                        if len(audio_chunk) == 0 or np.isnan(audio_chunk).any():
                            continue
                        
                        # Normalize audio to increase volume before writing to MP3
                        volume_multiplier = 2.0
                        # Apply higher volume for Chinese language MP3
                        if language == "ZH":
                            volume_multiplier = 8.47  # Equivalent to 13dB increase
                            
                        # Log volume multiplier value
                        print(f"Volume multiplier: {volume_multiplier:.1f}")
                            
                        normalized_audio = audio_chunk * volume_multiplier
                        # Clip to avoid distortion
                        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                        
                        sf.write(wav_io, normalized_audio, output_sample_rate, format="WAV")
                        wav_io.seek(0)
                        
                        # Convert to MP3 using FFmpeg (enhanced method from cosy-api.py)
                        try:
                            import subprocess
                            # Use FFmpeg to convert WAV to MP3 with optimized settings
                            process = subprocess.Popen(
                                [
                                    'ffmpeg',
                                    '-f', 's16le',  # 16-bit little-endian PCM
                                    '-ar', str(output_sample_rate),  # Input sample rate
                                    '-ac', '1',  # Mono
                                    '-i', 'pipe:0',  # Read from stdin
                                    '-c:a', 'libmp3lame',  # MP3 encoder
                                    '-b:a', '128k',  # 128 kbps
                                    '-q:a', '2',  # Quality setting
                                    '-write_id3v1', '0',  # No ID3v1
                                    '-write_id3v2', '0',  # No ID3v2
                                    '-id3v2_version', '0',  # No ID3v2
                                    '-write_xing', '0',  # No Xing header
                                    '-fflags', '+bitexact',
                                    '-f', 'mp3',  # MP3 format
                                    'pipe:1'  # Output to stdout
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )
                            
                            # Convert to PCM data for FFmpeg
                            normalized_pcm = (normalized_audio * 32767).astype(np.int16)
                            mp3_data, error = process.communicate(input=normalized_pcm.tobytes())
                            
                            if process.returncode != 0:
                                print(f"FFmpeg error: {error.decode()}")
                                continue
                            
                            # Trim MP3 padding for smoother playback (method from cosy-api.py)
                            trimmed_mp3_data = trim_mp3_padding(mp3_data)
                            
                            # Send MP3 chunk
                            await websocket.send_bytes(trimmed_mp3_data)
                            
                            # Save MP3 chunk if requested
                            if save_audio_files and chunk_save_folder:
                                chunk_filename = f"chunk_{chunk_counter}.mp3"
                                chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                try:
                                    with open(chunk_filepath, 'wb') as f:
                                        f.write(trimmed_mp3_data)
                                    print(f"[WS-TTS] Saved {chunk_filename} ({len(trimmed_mp3_data)} bytes)")
                                except Exception as save_error:
                                    print(f"[WS-TTS] Error saving chunk file: {save_error}")
                            
                            chunk_processing_time = (time.time() - chunk_start_time) * 1000
                            print(f"[WS-TTS] MP3 chunk {chunk_counter} sent: {len(trimmed_mp3_data)} bytes, time: {chunk_processing_time:.2f}ms")
                            
                        except Exception as e:
                            print(f"Enhanced MP3 conversion failed for chunk {chunk_counter}: {str(e)}")
                            continue
                    else:
                        await websocket.send_text(json.dumps({"error": f"Unsupported audio format: {audio_format}"}))
                        break
                    
                    chunk_counter += 1
                    await asyncio.sleep(0)  # Allow other tasks
                
                # Log completion
                generation_time = time.time() - start_time
                print(f"[WS-TTS] Audio streaming completed in {generation_time:.2f} seconds")
                print(f"[WS-TTS] Total chunks sent: {chunk_counter}")
                
                # Send an empty chunk to signal completion
                await websocket.send_bytes(b'')
                
            except Exception as e:
                print(f"Error in websocket_tts: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                await websocket.send_text(json.dumps({"error": f"Error generating audio: {str(e)}"}))
                
    except WebSocketDisconnect:
        print("[WS-TTS] WebSocket connection disconnected")
    except Exception as e:
        print(f"[WS-TTS] WebSocket error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MeloTTS API Server')
    parser.add_argument('--port', type=int, default=9003, help='Port number to run the server on (default: 9003)')
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
