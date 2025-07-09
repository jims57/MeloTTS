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

# Valid API Keys for WebSocket authentication
VALID_API_KEYS = {
    "sk-5z6y7x8w9v0u1t2s3r4q5p6o7n8m9l0k1j2i3h4g",
    "sk-3a4b5c6d7e8f9g0h1i2j3k4l5m6n7o8p9q0r1s2t",
    "sk-9m8n7b6v5c4x3z2a1s0d9f8g7h6j5k4l3p2o1i0u",
    "sk-7u6y5t4r3e2w1q0a9s8d7f6g5h4j3k2l1z0x9c8v"
}

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

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "MeloTTS API is running"}

@app.websocket("/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    print(f"[WS-TTS] WebSocket connection established")
    
    # Check X-API-Key header for authorization
    api_key = websocket.headers.get("x-api-key")
    if not api_key or api_key not in VALID_API_KEYS:
        print(f"[WS-TTS] Unauthorized access attempt with API key: {api_key}")
        await websocket.send_text(json.dumps({"error": "Unauthorized: Invalid or missing X-API-Key header"}))
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    print(f"[WS-TTS] Authorized connection with valid API key")
    
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
            language = request_data.get("language", "ZH")  # Extract language from JSON
            
            if not text:
                await websocket.send_text(json.dumps({"error": "Text is required"}))
                continue
            
            # Handle new English language codes and speaker mapping like coqui-api.py
            english_variants = ["en-au", "en-hk", "en-sg", "en-in", "en-us", "en-gb"]
            original_language = language.lower()
            
            # Map for MeloTTS speaker IDs based on language variants
            melo_speaker_id_map = {
                "en": 4,            # EN-Default (American accent) for default English
                "en-au": 3,         # EN-AU
                "en-hk": 0,         # EN-US (Chinese accent)
                "en-sg": 0,         # EN-US (Chinese accent)
                "en-in": 2,         # EN_INDIA
                "en-us": 4,         # EN-Default (American accent)
                "en-gb": 1          # EN-BR
            }
            
            # Default MeloTTS speaker ID if not in map
            melo_speaker_id = 0
            
            # Handle "zh" language code - convert it to "zh-cn" then to "ZH"
            if original_language == "zh":
                original_language = "zh-cn"
                print("Converted language code 'zh' to 'zh-cn'")
            
            # Check if it's an English variant or standard English
            if original_language in english_variants or original_language == "en":
                # Map the language variant to appropriate MeloTTS speaker ID
                melo_speaker_id = melo_speaker_id_map.get(original_language, 4)  # Default to EN-Default (4) if not in map
                # For MeloTTS, use uppercase "EN" for language
                language = "EN"
                # Override the speakerId with the mapped speaker ID
                speaker_id = melo_speaker_id
                print(f"Converted language '{original_language}' to '{language}' for TTS processing")
                print(f"Using MeloTTS speaker ID: {melo_speaker_id} for {original_language}")
            elif original_language == "zh-cn":
                language = "ZH"
                print(f"Converted language '{original_language}' to '{language}' for TTS processing")
            else:
                # For other languages, convert to uppercase
                language = original_language.upper()
            
            # Get the appropriate model - language is now properly processed
            if language not in global_models:
                await websocket.send_text(json.dumps({"error": f"Language '{language}' not supported"}))
                continue
            
            model = global_models[language]
            
            # Define punctuation markers for all supported MeloTTS languages
            punctuation_markers = [
                # English
                '.', '!', '?', ';', ',', ':',
                # Spanish
                '¡', '¿',
                # French
                '«', '»',
                # Chinese
                '。', '！', '？', '；', '，', '：', '、',
                # Japanese  
                # Korean (uses mostly English punctuation)
            ]
            
            # Function to split text by punctuation while keeping the punctuation
            def split_by_punctuation(text):
                segments = []
                current_segment = ""
                
                for char in text:
                    current_segment += char
                    if char in punctuation_markers:
                        if current_segment.strip():  # Only add non-empty segments
                            segments.append(current_segment.strip())
                        current_segment = ""
                
                # Add any remaining text
                if current_segment.strip():
                    segments.append(current_segment.strip())
                
                # If we have no splits (no punctuation in text), use the whole text
                if not segments:
                    segments = [text]
                
                # For Chinese text, ensure first segment isn't too long for fast response
                if language == "ZH" and segments and len(segments[0]) > 25:
                    # Extract a shorter first segment if it's Chinese and too long
                    # This helps get the first audio chunk to the client faster
                    first_part = segments[0][:25]
                    rest_part = segments[0][25:]
                    segments[0] = first_part
                    # Only insert the rest if it's not empty
                    if rest_part.strip():
                        segments.insert(1, rest_part)
                
                # Combine very short segments with the next segment for better quality
                combined_segments = []
                current_combined = ""
                
                for segment in segments:
                    # If current segment is short (less than 5 chars) or current_combined is empty
                    if len(segment) < 5 or not current_combined:
                        current_combined += " " + segment if current_combined else segment
                    else:
                        combined_segments.append(current_combined)
                        current_combined = segment
                
                # Add the last combined segment if it exists
                if current_combined:
                    combined_segments.append(current_combined)
                
                return combined_segments
            
            # Split the text into segments
            text_segments = split_by_punctuation(text)
            print(f"[WS-TTS] Split text into {len(text_segments)} segments by punctuation")
            for i, segment in enumerate(text_segments):
                print(f"[WS-TTS] Segment {i+1}: {segment[:50]}{'...' if len(segment) > 50 else ''}")
            
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
                
                before_inference_time = time.time()
                elapsed_since_start = (before_inference_time - start_time) * 1000
                print(f"[WS-TTS] 📊 Pre-processing time: {elapsed_since_start:.2f}ms")
                print(f"Time before inference: {elapsed_since_start:.2f} ms since start")
                
                # Add timing variables for first chunk tracking
                first_chunk_generated = False
                first_chunk_sent = False
                first_chunk_time = None
                first_chunk_since_request = None
                first_chunk_sent_since_request = None
                chunk_counter = 0
                
                # Process each text segment immediately
                for segment_idx, segment_text in enumerate(text_segments):
                    segment_start_time = time.time()
                    print(f"[WS-TTS] Processing segment {segment_idx+1}/{len(text_segments)}: {segment_text[:50]}{'...' if len(segment_text) > 50 else ''}")
                    
                    try:
                        # Generate audio for this segment
                        inference_start_time = time.time()
                        if segment_idx == 0:
                            print(f"[WS-TTS] 🚀 Starting first segment inference at: {time.strftime('%H:%M:%S.%f')[:-3]}")
                        
                        segment_audio = model.tts_to_file(
                            text=segment_text,
                            speaker_id=speaker_id,
                            output_path=None,  # Don't save to file
                            sdp_ratio=request_data.get("sdp_ratio", 0.2),
                            noise_scale=request_data.get("noise_scale", 0.6),
                            noise_scale_w=request_data.get("noise_scale_w", 0.8),
                            speed=request_data.get("speed", 1.0),
                            quiet=True
                        )
                        
                        # Record first chunk timing immediately after first segment inference
                        if not first_chunk_generated:
                            chunk_start_time = time.time()
                            first_chunk_time = (chunk_start_time - inference_start_time) * 1000
                            first_chunk_since_request = (chunk_start_time - start_time) * 1000
                            print(f"[WS-TTS] ⚡ First chunk generated time: {first_chunk_time:.2f}ms")
                            print(f"[WS-TTS] ⚡ First chunk since request arrival: {first_chunk_since_request:.2f}ms")
                            first_chunk_generated = True
                        
                        segment_inference_time = (time.time() - inference_start_time) * 1000
                        print(f"[WS-TTS] Segment {segment_idx+1} inference time: {segment_inference_time:.2f}ms")
                        
                        # Resample segment audio if needed
                        if model.hps.data.sampling_rate != output_sample_rate:
                            import torchaudio
                            resample_start = time.time()
                            audio_tensor = torch.from_numpy(segment_audio).unsqueeze(0)
                            audio_tensor = torchaudio.functional.resample(
                                audio_tensor, 
                                model.hps.data.sampling_rate, 
                                output_sample_rate
                            )
                            segment_audio = audio_tensor.squeeze(0).numpy()
                            resample_time = (time.time() - resample_start) * 1000
                            print(f"[WS-TTS] 🔄 Segment {segment_idx+1} resampled: {resample_time:.2f}ms ({model.hps.data.sampling_rate} → {output_sample_rate} Hz)")
                        else:
                            print(f"[WS-TTS] ✓ Segment {segment_idx+1}: No resampling needed (optimal)")
                        
                        # Stream segment audio in chunks
                        chunk_duration = 1.0  # 1 second chunks
                        samples_per_chunk = int(output_sample_rate * chunk_duration)
                        
                        # Stream audio data in chunks immediately
                        for i in range(0, len(segment_audio), samples_per_chunk):
                            chunk_start_time = time.time()
                            
                            # Extract audio chunk
                            audio_chunk = segment_audio[i:i + samples_per_chunk]
                            
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
                                
                                # Send PCM chunk immediately
                                await websocket.send_bytes(pcm_data)
                                
                                # Track first chunk sent timing
                                if not first_chunk_sent:
                                    first_chunk_sent_time = time.time()
                                    first_chunk_sent_since_request = (first_chunk_sent_time - start_time) * 1000
                                    print(f"[WS-TTS] 🎯 First chunk sent since request: {first_chunk_sent_since_request:.2f}ms")
                                    first_chunk_sent = True
                                
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
                                print(f"[WS-TTS] 📦 PCM chunk {chunk_counter} sent: {len(pcm_data)} bytes, time: {chunk_processing_time:.2f}ms")
                                
                            elif audio_format.lower() == "mp3":
                                # Convert chunk to MP3 with fallback when FFmpeg is not available
                                import soundfile as sf
                                
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
                                
                                # Try FFmpeg first, fallback to soundfile if FFmpeg not available
                                mp3_data = None
                                try:
                                    import subprocess
                                    import shutil
                                    
                                    # Check if ffmpeg is available
                                    if shutil.which('ffmpeg') is None:
                                        raise FileNotFoundError("FFmpeg not found")
                                    
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
                                        raise Exception("FFmpeg conversion failed")
                                    
                                    # Trim MP3 padding for smoother playback
                                    mp3_data = trim_mp3_padding(mp3_data)
                                    print(f"[WS-TTS] MP3 conversion via FFmpeg successful")
                                    
                                except Exception as ffmpeg_error:
                                    print(f"[WS-TTS] FFmpeg conversion failed: {str(ffmpeg_error)}")
                                    print(f"[WS-TTS] Falling back to WAV format for chunk {chunk_counter}")
                                    
                                    # Fallback: send as WAV format
                                    try:
                                        wav_io = io.BytesIO()
                                        sf.write(wav_io, normalized_audio, output_sample_rate, format="WAV")
                                        mp3_data = wav_io.getvalue()
                                        wav_io.close()
                                        print(f"[WS-TTS] WAV fallback conversion successful")
                                    except Exception as wav_error:
                                        print(f"[WS-TTS] WAV fallback also failed: {str(wav_error)}")
                                        continue
                                
                                if mp3_data:
                                    # Send MP3/WAV chunk immediately
                                    await websocket.send_bytes(mp3_data)
                                    
                                    # Track first chunk sent timing
                                    if not first_chunk_sent:
                                        first_chunk_sent_time = time.time()
                                        first_chunk_sent_since_request = (first_chunk_sent_time - start_time) * 1000
                                        print(f"[WS-TTS] 🎯 First chunk sent since request: {first_chunk_sent_since_request:.2f}ms")
                                        first_chunk_sent = True
                                    
                                    # Save MP3 chunk if requested
                                    if save_audio_files and chunk_save_folder:
                                        chunk_filename = f"chunk_{chunk_counter}.mp3"
                                        chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                        try:
                                            with open(chunk_filepath, 'wb') as f:
                                                f.write(mp3_data)
                                            print(f"[WS-TTS] Saved {chunk_filename} ({len(mp3_data)} bytes)")
                                        except Exception as save_error:
                                            print(f"[WS-TTS] Error saving chunk file: {save_error}")
                                    
                                    chunk_processing_time = (time.time() - chunk_start_time) * 1000
                                    print(f"[WS-TTS] 📦 MP3 chunk {chunk_counter} sent: {len(mp3_data)} bytes, time: {chunk_processing_time:.2f}ms")
                            
                            else:
                                await websocket.send_text(json.dumps({"error": f"Unsupported audio format: {audio_format}"}))
                                break
                            
                            chunk_counter += 1
                            await asyncio.sleep(0)  # Allow other tasks
                        
                        segment_total_time = (time.time() - segment_start_time) * 1000
                        print(f"[WS-TTS] 📊 Segment {segment_idx+1} total processing time: {segment_total_time:.2f}ms")
                        
                    except Exception as segment_error:
                        print(f"Error processing segment {segment_idx+1}: {str(segment_error)}")
                        continue
                
                # Log completion
                generation_time = time.time() - start_time
                print(f"[WS-TTS] 🏁 Audio streaming completed in {generation_time:.2f} seconds")
                print(f"[WS-TTS] Total chunks sent: {chunk_counter}")
                
                # === REQUEST SUMMARY (like cosy-api.py) ===
                print(f"[WS-TTS] 📋 REQUEST SUMMARY:")
                print(f"[WS-TTS] 📋   Audio Format: {audio_format}")
                print(f"[WS-TTS] 📋   Sample Rate: {output_sample_rate} Hz")
                print(f"[WS-TTS] 📋   Speaker ID: {speaker_id}")
                print(f"[WS-TTS] 📋   Total Generation Time: {generation_time:.2f}s")
                print(f"[WS-TTS] 📋   Text Segments: {len(text_segments)}")
                
                # Add first chunk timing summary
                if first_chunk_generated:
                    print(f"[WS-TTS] 📋   First Chunk Generated Time: {first_chunk_time:.2f}ms")
                    print(f"[WS-TTS] 📋   First Chunk Since Request: {first_chunk_since_request:.2f}ms")
                    if first_chunk_sent_since_request is not None:
                        print(f"[WS-TTS] 📋   First Chunk Sent Since Request: {first_chunk_sent_since_request:.2f}ms")
                else:
                    print(f"[WS-TTS] 📋   First Chunk: Not generated")
                
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

def trim_mp3_padding(mp3_data):
    """Remove padding bytes from the end of MP3 chunk to ensure clean frame boundaries"""
    if len(mp3_data) < 4:
        return mp3_data
    
    # Convert to bytearray for easier manipulation
    data = bytearray(mp3_data)
    original_length = len(data)
    
    # Look for repetitive padding patterns at the end
    # Common MP3 padding patterns: 0x55, 0xAA, 0x00, etc.
    padding_patterns = [0x55, 0xAA, 0x00]
    
    # Find the last non-padding byte
    end_pos = len(data)
    
    for pattern in padding_patterns:
        # Check if we have repetitive padding pattern at the end
        consecutive_count = 0
        pos = len(data) - 1
        
        # Count consecutive padding bytes from the end
        while pos >= 0 and data[pos] == pattern:
            consecutive_count += 1
            pos -= 1
        
        # If we found significant padding (more than 16 consecutive bytes)
        if consecutive_count > 16:
            potential_end = pos + 1
            if potential_end < end_pos:
                end_pos = potential_end
                print(f"[WS-TTS] Detected {consecutive_count} bytes of 0x{pattern:02X} padding, trimming to position {end_pos}")
    
    # Additional check: look for MP3 frame sync patterns to avoid cutting in the middle of frames
    # MP3 frame sync is 0xFFF (first 11 bits), so we look for 0xFF followed by 0xF*
    if end_pos < original_length:
        # Try to align to the last valid MP3 frame boundary
        for i in range(end_pos - 1, max(0, end_pos - 100), -1):  # Look back up to 100 bytes
            if i + 1 < len(data) and data[i] == 0xFF and (data[i + 1] & 0xF0) == 0xF0:
                # Found potential MP3 frame sync, this might be a better cut point
                # Look for the end of this frame
                frame_start = i
                # MP3 frame header is 4 bytes, try to find frame length
                if frame_start + 4 <= len(data):
                    # For now, just cut here as it's a frame boundary
                    end_pos = min(end_pos, frame_start + 4)
                    print(f"[WS-TTS] Aligned to MP3 frame boundary at position {end_pos}")
                    break
    
    # Trim the data
    trimmed_data = bytes(data[:end_pos])
    
    if end_pos < original_length:
        bytes_removed = original_length - end_pos
        print(f"[WS-TTS] Trimmed {bytes_removed} padding bytes from MP3 chunk ({original_length} -> {end_pos} bytes)")
    
    return trimmed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MeloTTS API Server')
    parser.add_argument('--port', type=int, default=9003, help='Port number to run the server on (default: 9003)')
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
