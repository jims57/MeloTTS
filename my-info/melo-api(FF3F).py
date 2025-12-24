"""
# Author: Jimmy Gan
# Date: Dec 23, 2025
# Melo TTS API Server
# Version: 1.3.6
# Changes number: 5
"""
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
    
    def create_header_bytes(start_time_id, message_id):
        """
        创建固定长度的消息头部字节
        startTimeId: 8字节 (64位大端序整数)
        messageId: 4字节 (32位大端序整数)
        总计: 12字节
        """
        import struct
        # 使用大端序格式，便于Java解析
        # Q = 64位无符号整数，I = 32位无符号整数
        header = struct.pack('>QI', start_time_id, message_id)
        return header
    
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
            
            # 提取新增的消息标识参数
            start_time_id = request_data.get("startTimeId")
            message_id = request_data.get("messageId")
            
            # 判断是否需要添加消息头部
            has_message_headers = start_time_id is not None and message_id is not None
            
            if has_message_headers:
                print(f"[WS-TTS] Message headers detected - startTimeId: {start_time_id}, messageId: {message_id}")
                # 验证参数范围
                if not isinstance(start_time_id, int) or start_time_id < 0:
                    await websocket.send_text(json.dumps({"error": "startTimeId must be a non-negative integer"}))
                    continue
                if not isinstance(message_id, int) or message_id < 1 or message_id > 4294967295:
                    await websocket.send_text(json.dumps({"error": "messageId must be an integer between 1 and 4294967295"}))
                    continue
            else:
                print(f"[WS-TTS] No message headers - using standard PCM streaming")
            
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
                                
                                # 根据是否有消息头部决定发送格式
                                if has_message_headers:
                                    # 添加消息头部到PCM数据前面
                                    header_bytes = create_header_bytes(start_time_id, message_id)
                                    data_to_send = header_bytes + pcm_data
                                    await websocket.send_bytes(data_to_send)
                                    print(f"[WS-TTS] 📦 PCM chunk {chunk_counter} sent with header: {len(header_bytes)} header + {len(pcm_data)} PCM = {len(data_to_send)} total bytes")
                                else:
                                    # 直接发送PCM数据
                                    await websocket.send_bytes(pcm_data)
                                    print(f"[WS-TTS] 📦 PCM chunk {chunk_counter} sent: {len(pcm_data)} bytes")
                                
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
                                            # 保存与发送给客户端相同的数据格式
                                            if has_message_headers:
                                                # 保存带头部的数据
                                                header_bytes = create_header_bytes(start_time_id, message_id)
                                                f.write(header_bytes + pcm_data)
                                                print(f"[WS-TTS] Saved {chunk_filename} with header ({len(header_bytes + pcm_data)} bytes)")
                                            else:
                                                # 保存纯PCM数据
                                                f.write(pcm_data)
                                                print(f"[WS-TTS] Saved {chunk_filename} ({len(pcm_data)} bytes)")
                                    except Exception as save_error:
                                        print(f"[WS-TTS] Error saving chunk file: {save_error}")
                                
                                chunk_processing_time = (time.time() - chunk_start_time) * 1000
                                print(f"[WS-TTS] Send time: {chunk_processing_time:.2f}ms")
                                
                            elif audio_format.lower() == "mp3":
                                # Gapless MP3 streaming: CBR encoding with encoder delay handling
                                # Based on ai-2.txt: pre-padding + clipping strategy for seamless concatenation
                                import soundfile as sf
                                import subprocess
                                import shutil
                                
                                # Check if audio is valid
                                if len(audio_chunk) == 0 or np.isnan(audio_chunk).any():
                                    continue
                                
                                # Volume adjustment
                                volume_multiplier = 2.0
                                if language == "ZH":
                                    volume_multiplier = 8.47  # +13dB for Chinese
                                    
                                print(f"Volume multiplier: {volume_multiplier:.1f}")
                                    
                                normalized_audio = audio_chunk * volume_multiplier
                                normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                                
                                # LAME encoder delay: 576 samples for 16kHz
                                # Pre-padding: add silence at the beginning to compensate encoder delay
                                encoder_delay_samples = 576
                                silence_padding = np.zeros(encoder_delay_samples, dtype=np.float32)
                                padded_audio = np.concatenate([silence_padding, normalized_audio])
                                
                                mp3_data = None
                                try:
                                    if shutil.which('ffmpeg') is None:
                                        raise FileNotFoundError("FFmpeg not found")
                                    
                                    # FFmpeg with CBR encoding for gapless playback
                                    # Key settings:
                                    # - CBR mode (-b:a 128k) for consistent frame size
                                    # - No metadata (-write_xing 0, -id3v2_version 0)
                                    # - Bit-exact output (-fflags +bitexact)
                                    process = subprocess.Popen(
                                        [
                                            'ffmpeg',
                                            '-f', 's16le',
                                            '-ar', str(output_sample_rate),
                                            '-ac', '1',
                                            '-i', 'pipe:0',
                                            '-c:a', 'libmp3lame',
                                            '-b:a', '128k',  # CBR 128kbps
                                            '-write_xing', '0',
                                            '-id3v2_version', '0',
                                            '-fflags', '+bitexact',
                                            '-f', 'mp3',
                                            'pipe:1'
                                        ],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE
                                    )
                                    
                                    # Convert padded audio to PCM
                                    padded_pcm = (padded_audio * 32767).astype(np.int16)
                                    mp3_data, error = process.communicate(input=padded_pcm.tobytes())
                                    
                                    if process.returncode != 0:
                                        print(f"FFmpeg error: {error.decode()}")
                                        raise Exception("FFmpeg conversion failed")
                                    
                                    # Clipping: remove the first MP3 frame(s) that contain encoder delay
                                    # MP3 frame at 128kbps, 16kHz, mono = 576 samples per frame
                                    # Frame size = 576 * 128000 / (8 * 16000) = 576 bytes per frame
                                    # We need to skip frames containing the pre-padded silence
                                    mp3_data = strip_encoder_delay_frames(mp3_data, encoder_delay_samples, output_sample_rate)
                                    
                                    print(f"[WS-TTS] Gapless MP3 encoded: {len(mp3_data)} bytes")
                                    
                                except Exception as ffmpeg_error:
                                    print(f"[WS-TTS] FFmpeg error: {str(ffmpeg_error)}")
                                    print(f"[WS-TTS] Falling back to WAV for chunk {chunk_counter}")
                                    
                                    try:
                                        wav_io = io.BytesIO()
                                        sf.write(wav_io, normalized_audio, output_sample_rate, format="WAV")
                                        mp3_data = wav_io.getvalue()
                                        wav_io.close()
                                        print(f"[WS-TTS] WAV fallback successful")
                                    except Exception as wav_error:
                                        print(f"[WS-TTS] WAV fallback failed: {str(wav_error)}")
                                        continue
                                
                                if mp3_data:
                                    await websocket.send_bytes(mp3_data)
                                    
                                    if not first_chunk_sent:
                                        first_chunk_sent_time = time.time()
                                        first_chunk_sent_since_request = (first_chunk_sent_time - start_time) * 1000
                                        print(f"[WS-TTS] First chunk sent: {first_chunk_sent_since_request:.2f}ms")
                                        first_chunk_sent = True
                                    
                                    if save_audio_files and chunk_save_folder:
                                        chunk_filename = f"chunk_{chunk_counter}.mp3"
                                        chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                        try:
                                            with open(chunk_filepath, 'wb') as f:
                                                f.write(mp3_data)
                                            print(f"[WS-TTS] Saved {chunk_filename} ({len(mp3_data)} bytes)")
                                        except Exception as save_error:
                                            print(f"[WS-TTS] Error saving chunk: {save_error}")
                                    
                                    chunk_processing_time = (time.time() - chunk_start_time) * 1000
                                    print(f"[WS-TTS] MP3 chunk {chunk_counter}: {len(mp3_data)} bytes, {chunk_processing_time:.2f}ms")
                            
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
                print(f"[WS-TTS] 📋   Message Headers: {'Yes' if has_message_headers else 'No'}")
                if has_message_headers:
                    print(f"[WS-TTS] 📋   StartTimeId: {start_time_id}")
                    print(f"[WS-TTS] 📋   MessageId: {message_id}")
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
                if has_message_headers:
                    # 发送空的完成信号时也要添加消息头部
                    header_bytes = create_header_bytes(start_time_id, message_id)
                    completion_signal = header_bytes + b''
                    await websocket.send_bytes(completion_signal)
                    print(f"[WS-TTS] 📡 Empty completion chunk sent with header for {audio_format.upper()} format")
                else:
                    # 标准的空完成信号
                    await websocket.send_bytes(b'')
                    print(f"[WS-TTS] 📡 Empty completion chunk sent for {audio_format.upper()} format")
                
            except Exception as e:
                print(f"Error in websocket_tts: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                await websocket.send_text(json.dumps({"error": f"Error generating audio: {str(e)}"}))
                
    except WebSocketDisconnect:
        print("[WS-TTS] WebSocket connection disconnected")
    except Exception as e:
        print(f"[WS-TTS] WebSocket error: {str(e)}")

def strip_encoder_delay_frames(mp3_data, delay_samples, sample_rate):
    """
    Strip MP3 frames containing encoder delay (pre-padded silence).
    For gapless playback, we need to remove the frames that contain
    the silence we added to compensate for LAME encoder delay.
    
    MP3 frame structure:
    - Frame sync: 0xFF 0xFB (or 0xFF 0xFA for MPEG1 Layer3)
    - Each frame contains 576 samples for MPEG1 Layer3
    - Frame size at 128kbps, 16kHz = 576 * 128000 / (8 * 16000) = 576 bytes
    """
    if len(mp3_data) < 4:
        return mp3_data
    
    data = bytearray(mp3_data)
    
    # Calculate how many frames to skip based on delay samples
    # MPEG1 Layer3: 576 samples per frame
    samples_per_frame = 576
    frames_to_skip = (delay_samples + samples_per_frame - 1) // samples_per_frame
    
    # Find and skip the first N frames
    pos = 0
    frames_skipped = 0
    
    while pos < len(data) - 4 and frames_skipped < frames_to_skip:
        # Look for frame sync: 0xFF followed by 0xFx (where x has bit 4 set)
        if data[pos] == 0xFF and (data[pos + 1] & 0xE0) == 0xE0:
            # Found potential frame header, calculate frame size
            # Header format: AAAAAAAA AAABBCCD EEEEFFGH IIJJKLMM
            # B = MPEG version, C = Layer, E = Bitrate index, F = Sample rate index
            header = (data[pos] << 24) | (data[pos+1] << 16) | (data[pos+2] << 8) | data[pos+3]
            
            # Extract fields
            version = (header >> 19) & 0x03  # 00=2.5, 01=reserved, 10=2, 11=1
            layer = (header >> 17) & 0x03    # 00=reserved, 01=III, 10=II, 11=I
            bitrate_idx = (header >> 12) & 0x0F
            srate_idx = (header >> 10) & 0x03
            padding = (header >> 9) & 0x01
            
            # Bitrate table for MPEG1 Layer3 (kbps)
            bitrates_v1_l3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
            # Bitrate table for MPEG2/2.5 Layer3 (kbps)
            bitrates_v2_l3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
            
            # Sample rate table
            srates_v1 = [44100, 48000, 32000, 0]
            srates_v2 = [22050, 24000, 16000, 0]
            srates_v25 = [11025, 12000, 8000, 0]
            
            # Determine frame size
            if version == 3:  # MPEG1
                bitrate = bitrates_v1_l3[bitrate_idx] * 1000
                srate = srates_v1[srate_idx]
                frame_samples = 1152 if layer == 1 else 576  # Layer I vs Layer II/III
            elif version == 2:  # MPEG2
                bitrate = bitrates_v2_l3[bitrate_idx] * 1000
                srate = srates_v2[srate_idx]
                frame_samples = 576
            else:  # MPEG2.5
                bitrate = bitrates_v2_l3[bitrate_idx] * 1000
                srate = srates_v25[srate_idx]
                frame_samples = 576
            
            if bitrate > 0 and srate > 0:
                # Frame size formula for Layer III
                if version == 3:  # MPEG1
                    frame_size = (144 * bitrate // srate) + padding
                else:  # MPEG2/2.5
                    frame_size = (72 * bitrate // srate) + padding
                
                # Skip this frame
                pos += frame_size
                frames_skipped += 1
            else:
                pos += 1
        else:
            pos += 1
    
    # Return data starting from after skipped frames
    if frames_skipped > 0:
        print(f"[WS-TTS] Stripped {frames_skipped} encoder delay frames ({pos} bytes)")
    
    return bytes(data[pos:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MeloTTS API Server')
    parser.add_argument('--port', type=int, default=9003, help='Port number to run the server on (default: 9003)')
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
