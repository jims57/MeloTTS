"""
# Author: Jimmy Gan
# Date: Dec 23, 2025
# Melo TTS API Server
# Version: 1.3.6
# Changes number:38
# head -n 7 melo-api.py
# cd ~/MeloTTS &&/root/MeloTTS/melotts/bin/python melo-api.py --port 9001
"""
import torch
import numpy as np
import io
import json
import base64
import argparse
import sys
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import time
from melo.api import TTS
import os
import asyncio

# 确保日志立即输出，不缓冲
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
            chunk_duration = request_data.get("chunkDuration", 0.25)  # 客户端可控制MP3 chunk时长，默认0.25秒
            
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
                
                # 创建持久的MP3编码器（用于无缝MP3流）
                mp3_encoder = None
                if audio_format.lower() == "mp3":
                    mp3_encoder = GaplessMP3Encoder(sample_rate=output_sample_rate, bitrate='128k')
                    print(f"[WS-TTS] Created GaplessMP3Encoder for seamless streaming")
                
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
                        # 使用客户端指定的chunk_duration，默认0.25秒
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
                                # 使用持久的GaplessMP3Encoder进行无缝编码
                                import soundfile as sf
                                
                                # Check if audio is valid
                                if len(audio_chunk) == 0 or np.isnan(audio_chunk).any():
                                    continue
                                
                                # 设置音量倍数
                                volume_multiplier = 2.0
                                if language == "ZH":
                                    volume_multiplier = 8.47  # Equivalent to 13dB increase
                                
                                mp3_data = None
                                try:
                                    # 使用持久编码器进行无缝编码
                                    mp3_data = mp3_encoder.feed_pcm(audio_chunk, volume_multiplier)
                                    
                                    if mp3_data:
                                        print(f"[WS-TTS] GaplessMP3: fed {len(audio_chunk)} samples, got {len(mp3_data)} bytes")
                                    else:
                                        print(f"[WS-TTS] GaplessMP3: fed {len(audio_chunk)} samples, buffering...")
                                        continue  # 缓冲中，等待更多数据
                                    
                                except Exception as encoder_error:
                                    print(f"[WS-TTS] GaplessMP3 encoder error: {str(encoder_error)}")
                                    
                                    # Fallback: send as WAV format
                                    try:
                                        wav_io = io.BytesIO()
                                        normalized_audio = np.clip(audio_chunk * volume_multiplier, -1.0, 1.0)
                                        sf.write(wav_io, normalized_audio, output_sample_rate, format="WAV")
                                        mp3_data = wav_io.getvalue()
                                        wav_io.close()
                                        print(f"[WS-TTS] WAV fallback conversion successful")
                                    except Exception as wav_error:
                                        print(f"[WS-TTS] WAV fallback also failed: {str(wav_error)}")
                                        continue
                                
                                if mp3_data:
                                    # 根据是否有消息头部决定发送格式
                                    if has_message_headers:
                                        # 添加消息头部到MP3数据前面
                                        header_bytes = create_header_bytes(start_time_id, message_id)
                                        data_to_send = header_bytes + mp3_data
                                        await websocket.send_bytes(data_to_send)
                                        print(f"[WS-TTS] 📦 MP3 chunk {chunk_counter} sent with header: {len(header_bytes)} header + {len(mp3_data)} MP3 = {len(data_to_send)} total bytes")
                                    else:
                                        # 直接发送MP3数据
                                        await websocket.send_bytes(mp3_data)
                                        print(f"[WS-TTS] 📦 MP3 chunk {chunk_counter} sent: {len(mp3_data)} bytes")
                                    
                                    # Track first chunk sent timing
                                    if not first_chunk_sent:
                                        first_chunk_sent_time = time.time()
                                        first_chunk_sent_since_request = (first_chunk_sent_time - start_time) * 1000
                                        print(f"[WS-TTS] First chunk sent since request: {first_chunk_sent_since_request:.2f}ms")
                                        first_chunk_sent = True
                                    
                                    # Save MP3 chunk if requested
                                    if save_audio_files and chunk_save_folder:
                                        chunk_filename = f"chunk_{chunk_counter}.mp3"
                                        chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                        try:
                                            with open(chunk_filepath, 'wb') as f:
                                                # 保存与发送给客户端相同的数据格式
                                                if has_message_headers:
                                                    # 保存带头部的数据
                                                    header_bytes = create_header_bytes(start_time_id, message_id)
                                                    f.write(header_bytes + mp3_data)
                                                    print(f"[WS-TTS] Saved {chunk_filename} with header ({len(header_bytes + mp3_data)} bytes)")
                                                else:
                                                    # 保存纯MP3数据
                                                    f.write(mp3_data)
                                                    print(f"[WS-TTS] Saved {chunk_filename} ({len(mp3_data)} bytes)")
                                        except Exception as save_error:
                                            print(f"[WS-TTS] Error saving chunk file: {save_error}")
                                    
                                    chunk_processing_time = (time.time() - chunk_start_time) * 1000
                            
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
                
                # 刷新MP3编码器，获取剩余数据
                if mp3_encoder is not None:
                    try:
                        final_mp3_data = mp3_encoder.flush()
                        if final_mp3_data:
                            if has_message_headers:
                                header_bytes = create_header_bytes(start_time_id, message_id)
                                data_to_send = header_bytes + final_mp3_data
                                await websocket.send_bytes(data_to_send)
                                print(f"[WS-TTS] GaplessMP3: flushed final {len(final_mp3_data)} bytes with header")
                            else:
                                await websocket.send_bytes(final_mp3_data)
                                print(f"[WS-TTS] GaplessMP3: flushed final {len(final_mp3_data)} bytes")
                            chunk_counter += 1
                        mp3_encoder.close()
                    except Exception as flush_error:
                        print(f"[WS-TTS] GaplessMP3 flush error: {str(flush_error)}")
                
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

def is_xing_frame(data, pos):
    """
    检查指定位置的帧是否为Xing/Info帧 (VBR元数据帧)
    
    Args:
        data: MP3数据
        pos: 帧起始位置
    
    Returns:
        True如果是Xing/Info帧
    """
    search_range = data[pos:pos+200]
    return b'Xing' in search_range or b'Info' in search_range or b'LAME' in search_range


def find_lame_tag_position(data):
    """
    查找LAME标签在数据中的位置
    LAME编码器会在MP3数据末尾嵌入"LAME3.100"等标签，后面跟着填充字节
    
    Args:
        data: MP3数据
    
    Returns:
        LAME标签的位置，如果未找到则返回-1
    """
    # 搜索LAME标签
    lame_pos = data.find(b'LAME')
    if lame_pos != -1:
        return lame_pos
    
    # 搜索Xing标签
    xing_pos = data.find(b'Xing')
    if xing_pos != -1:
        return xing_pos
    
    # 搜索Info标签
    info_pos = data.find(b'Info')
    if info_pos != -1:
        return info_pos
    
    return -1


def extract_raw_mp3_frames(mp3_data, skip_encoder_delay_frames=1):
    """
    从MP3数据中提取纯净的音频帧，用于无缝拼接
    - 跳过ID3头部、Xing/Info/LAME元数据帧
    - 跳过编码器延迟帧
    - 只保留有效的MP3音频帧
    - 截断末尾的LAME标签及填充字节
    
    Args:
        mp3_data: 原始MP3数据
        skip_encoder_delay_frames: 跳过开头的帧数（用于处理编码器延迟）
    
    Returns:
        纯净的MP3音频帧数据
    """
    if len(mp3_data) < 4:
        return mp3_data
    
    data = bytes(mp3_data)
    pos = 0
    
    # 步骤1: 跳过ID3v2头部
    if data[:3] == b'ID3' and len(data) >= 10:
        id3_size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | \
                   ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        pos = 10 + id3_size
        print(f"[WS-TTS] Skipped ID3v2 header: {pos} bytes")
    
    # 步骤2: 查找末尾LAME标签位置，确定有效数据结束位置
    end_pos = len(data)
    lame_pos = find_lame_tag_position(data)
    if lame_pos != -1 and lame_pos > pos:
        # 向前搜索LAME标签所在帧的起始位置
        frame_start = lame_pos
        while frame_start > pos:
            if data[frame_start] == 0xFF and frame_start + 1 < len(data) and (data[frame_start + 1] & 0xE0) == 0xE0:
                break
            frame_start -= 1
        
        if frame_start > pos:
            end_pos = frame_start
            print(f"[WS-TTS] Truncated at LAME tag, end_pos: {end_pos} (original: {len(data)})")
    
    frames = []
    frame_count = 0
    skipped_delay_frames = 0
    skipped_xing_frames = 0
    
    while pos < end_pos - 4:
        # 查找MP3帧同步字
        if data[pos] == 0xFF and (data[pos + 1] & 0xE0) == 0xE0:
            # 解析MP3帧头
            header = (data[pos] << 24) | (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3]
            
            version = (header >> 19) & 0x03
            layer = (header >> 17) & 0x03
            bitrate_index = (header >> 12) & 0x0F
            sample_rate_index = (header >> 10) & 0x03
            padding = (header >> 9) & 0x01
            
            # 验证帧头有效性
            if layer == 0 or bitrate_index == 0 or bitrate_index == 15 or sample_rate_index == 3:
                pos += 1
                continue
            
            frame_length = calculate_mp3_frame_length(version, layer, bitrate_index, sample_rate_index, padding)
            
            if frame_length > 0 and pos + frame_length <= end_pos:
                # 检查是否为Xing/Info/LAME元数据帧
                if is_xing_frame(data, pos):
                    skipped_xing_frames += 1
                    pos += frame_length
                    continue
                
                # 跳过编码器延迟帧
                if skipped_delay_frames < skip_encoder_delay_frames:
                    skipped_delay_frames += 1
                    pos += frame_length
                    continue
                
                # 验证下一帧也是有效的MP3帧（双重验证）
                next_pos = pos + frame_length
                if next_pos < end_pos - 4:
                    next_header = data[next_pos:next_pos+4]
                    if len(next_header) >= 4 and next_header[0] == 0xFF and (next_header[1] & 0xE0) == 0xE0:
                        # 下一帧也有效，保存当前帧
                        frames.append(data[pos:pos + frame_length])
                        frame_count += 1
                    else:
                        # 下一帧无效，可能是最后一帧或损坏数据，跳过
                        pass
                else:
                    # 文件末尾，不保存最后一帧（可能包含填充）
                    pass
                
                pos += frame_length
            else:
                pos += 1
        else:
            pos += 1
    
    if frame_count > 0:
        print(f"[WS-TTS] Extracted {frame_count} MP3 frames, skipped {skipped_delay_frames} delay + {skipped_xing_frames} xing frames")
    
    return b''.join(frames)


def calculate_mp3_frame_length(version, layer, bitrate_index, sample_rate_index, padding):
    """
    计算MP3帧长度
    
    Args:
        version: MPEG版本 (0=2.5, 2=2, 3=1)
        layer: Layer (1=III, 2=II, 3=I)
        bitrate_index: 比特率索引
        sample_rate_index: 采样率索引
        padding: 填充位
    
    Returns:
        帧长度（字节）
    """
    # 比特率表（kbps）
    # MPEG-1 Layer III
    bitrate_table_v1_l3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    # MPEG-2/2.5 Layer III
    bitrate_table_v2_l3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
    
    # 采样率表（Hz）
    sample_rate_table = {
        3: [44100, 48000, 32000],  # MPEG-1
        2: [22050, 24000, 16000],  # MPEG-2
        0: [11025, 12000, 8000]    # MPEG-2.5
    }
    
    # 获取比特率
    if version == 3:  # MPEG-1
        bitrate = bitrate_table_v1_l3[bitrate_index] * 1000
    else:  # MPEG-2 or MPEG-2.5
        bitrate = bitrate_table_v2_l3[bitrate_index] * 1000
    
    # 获取采样率
    if version not in sample_rate_table:
        return 0
    if sample_rate_index >= len(sample_rate_table[version]):
        return 0
    sample_rate = sample_rate_table[version][sample_rate_index]
    
    if bitrate == 0 or sample_rate == 0:
        return 0
    
    # Layer III帧长度计算公式
    # 对于MPEG-1: frame_length = 144 * bitrate / sample_rate + padding
    # 对于MPEG-2/2.5: frame_length = 72 * bitrate / sample_rate + padding
    if version == 3:  # MPEG-1
        frame_length = (144 * bitrate) // sample_rate + padding
    else:  # MPEG-2 or MPEG-2.5
        frame_length = (72 * bitrate) // sample_rate + padding
    
    return frame_length


def align_pcm_to_mp3_frame_size(pcm_samples, sample_rate):
    """
    将PCM样本数对齐到MP3帧大小（1152样本）
    这是实现无缝MP3拼接的关键
    
    Args:
        pcm_samples: PCM样本数组
        sample_rate: 采样率
    
    Returns:
        对齐后的PCM样本数组
    """
    # MP3帧包含1152个样本（MPEG-1 Layer III）
    MP3_FRAME_SAMPLES = 1152
    
    current_samples = len(pcm_samples)
    
    # 计算需要的样本数（向上取整到1152的整数倍）
    aligned_samples = ((current_samples + MP3_FRAME_SAMPLES - 1) // MP3_FRAME_SAMPLES) * MP3_FRAME_SAMPLES
    
    if aligned_samples > current_samples:
        # 添加静音样本进行填充
        padding_samples = aligned_samples - current_samples
        padded_pcm = np.concatenate([pcm_samples, np.zeros(padding_samples, dtype=pcm_samples.dtype)])
        print(f"[WS-TTS] Aligned PCM from {current_samples} to {aligned_samples} samples (+{padding_samples} padding)")
        return padded_pcm
    
    return pcm_samples


class GaplessMP3Encoder:
    """
    无缝MP3编码器 - 保持单一FFmpeg进程，实现真正的无缝MP3流
    
    关键特性:
    1. 保持单一编码器实例，维护内部状态一致性
    2. PCM缓冲区处理帧边界对齐（1152样本）
    3. 只输出纯净的MP3音频帧，无元数据
    """
    
    MP3_FRAME_SAMPLES = 1152  # MPEG-1 Layer III每帧样本数
    
    def __init__(self, sample_rate=24000, bitrate='128k'):
        self.sample_rate = sample_rate
        self.bitrate = bitrate
        self.process = None
        self.pcm_buffer = np.array([], dtype=np.float32)  # PCM缓冲区
        self.mp3_buffer = b''  # MP3输出缓冲区
        self.is_first_chunk = True
        self.frames_written = 0
        
    def start(self):
        """启动FFmpeg编码器进程"""
        import subprocess
        import shutil
        
        if shutil.which('ffmpeg') is None:
            raise FileNotFoundError("FFmpeg not found")
        
        # 启动持久的FFmpeg进程
        self.process = subprocess.Popen(
            [
                'ffmpeg',
                '-f', 's16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                '-i', 'pipe:0',
                '-c:a', 'libmp3lame',
                '-b:a', self.bitrate,
                '-q:a', '2',
                '-write_id3v1', '0',
                '-write_id3v2', '0',
                '-id3v2_version', '0',
                '-write_xing', '0',
                '-fflags', '+bitexact',
                '-f', 'mp3',
                'pipe:1'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0  # 无缓冲，实时输出
        )
        print(f"[GaplessMP3] Encoder started: {self.sample_rate}Hz, {self.bitrate}")
        
    def feed_pcm(self, pcm_float, volume_multiplier=1.0):
        """
        输入PCM数据，返回可用的MP3帧
        
        Args:
            pcm_float: float32 PCM数据 [-1.0, 1.0]
            volume_multiplier: 音量倍数
        
        Returns:
            bytes: 纯净的MP3音频帧数据
        """
        if self.process is None:
            self.start()
        
        # 应用音量并裁剪
        audio = pcm_float * volume_multiplier
        audio = np.clip(audio, -1.0, 1.0)
        
        # 添加到PCM缓冲区
        self.pcm_buffer = np.concatenate([self.pcm_buffer, audio])
        
        # 计算可以编码的完整帧数
        complete_frames = len(self.pcm_buffer) // self.MP3_FRAME_SAMPLES
        
        if complete_frames == 0:
            return b''  # 缓冲区不足一帧
        
        # 取出完整帧的样本
        samples_to_encode = complete_frames * self.MP3_FRAME_SAMPLES
        pcm_to_encode = self.pcm_buffer[:samples_to_encode]
        self.pcm_buffer = self.pcm_buffer[samples_to_encode:]  # 保留剩余样本
        
        # 转换为16位PCM并写入编码器
        pcm_int16 = (pcm_to_encode * 32767).astype(np.int16)
        self.process.stdin.write(pcm_int16.tobytes())
        self.process.stdin.flush()
        
        # 非阻塞读取MP3输出
        import select
        mp3_output = b''
        
        while True:
            # 检查是否有数据可读（非阻塞）
            readable, _, _ = select.select([self.process.stdout], [], [], 0.01)
            if not readable:
                break
            
            chunk = self.process.stdout.read(4096)
            if not chunk:
                break
            mp3_output += chunk
        
        if mp3_output:
            # 提取纯净的MP3帧
            clean_frames = self._extract_audio_frames(mp3_output)
            self.frames_written += 1
            return clean_frames
        
        return b''
    
    def flush(self):
        """
        刷新编码器，获取剩余的MP3数据
        
        Returns:
            bytes: 剩余的MP3帧数据
        """
        if self.process is None:
            return b''
        
        # 如果缓冲区还有数据，用静音填充到完整帧
        if len(self.pcm_buffer) > 0:
            padding_needed = self.MP3_FRAME_SAMPLES - (len(self.pcm_buffer) % self.MP3_FRAME_SAMPLES)
            if padding_needed < self.MP3_FRAME_SAMPLES:
                self.pcm_buffer = np.concatenate([self.pcm_buffer, np.zeros(padding_needed, dtype=np.float32)])
            
            pcm_int16 = (self.pcm_buffer * 32767).astype(np.int16)
            self.process.stdin.write(pcm_int16.tobytes())
            self.pcm_buffer = np.array([], dtype=np.float32)
        
        # 关闭stdin触发编码器刷新
        self.process.stdin.close()
        
        # 读取所有剩余输出
        mp3_output = self.process.stdout.read()
        self.process.wait()
        
        if mp3_output:
            # 提取纯净帧，但跳过最后可能包含LAME标签的帧
            clean_frames = self._extract_audio_frames(mp3_output, skip_last=True)
            return clean_frames
        
        return b''
    
    def _extract_audio_frames(self, mp3_data, skip_last=False):
        """
        从MP3数据中提取纯净的音频帧
        
        Args:
            mp3_data: 原始MP3数据
            skip_last: 是否跳过最后一帧（可能包含LAME标签）
        
        Returns:
            bytes: 纯净的MP3音频帧
        """
        if len(mp3_data) < 4:
            return mp3_data
        
        data = bytes(mp3_data)
        pos = 0
        frames = []
        
        # 跳过ID3头部
        if data[:3] == b'ID3' and len(data) >= 10:
            id3_size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | \
                       ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
            pos = 10 + id3_size
        
        while pos < len(data) - 4:
            if data[pos] == 0xFF and (data[pos + 1] & 0xE0) == 0xE0:
                header = (data[pos] << 24) | (data[pos + 1] << 16) | (data[pos + 2] << 8) | data[pos + 3]
                
                version = (header >> 19) & 0x03
                layer = (header >> 17) & 0x03
                bitrate_index = (header >> 12) & 0x0F
                sample_rate_index = (header >> 10) & 0x03
                padding = (header >> 9) & 0x01
                
                if layer == 0 or bitrate_index == 0 or bitrate_index == 15 or sample_rate_index == 3:
                    pos += 1
                    continue
                
                frame_length = calculate_mp3_frame_length(version, layer, bitrate_index, sample_rate_index, padding)
                
                if frame_length > 0 and pos + frame_length <= len(data):
                    # 检查是否为Xing/Info/LAME元数据帧
                    frame_content = data[pos:pos+200]
                    if b'Xing' in frame_content or b'Info' in frame_content or b'LAME' in frame_content:
                        pos += frame_length
                        continue
                    
                    frames.append((pos, frame_length))
                    pos += frame_length
                else:
                    pos += 1
            else:
                pos += 1
        
        # 如果需要跳过最后一帧
        if skip_last and len(frames) > 0:
            frames = frames[:-1]
        
        # 组装纯净帧
        result = b''
        for frame_pos, frame_len in frames:
            result += data[frame_pos:frame_pos + frame_len]
        
        return result
    
    def close(self):
        """关闭编码器"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.stdout.close()
                self.process.stderr.close()
                self.process.terminate()
                self.process.wait(timeout=1)
            except:
                pass
            self.process = None
        print(f"[GaplessMP3] Encoder closed, total writes: {self.frames_written}")

@app.get("/apiDoc")
async def get_api_documentation():
    """
    API Documentation Endpoint
    
    Returns:
    - HTML response with API documentation
    """
    try:
        # 获取API文档HTML文件路径
        html_file_path = os.path.join(os.path.dirname(__file__), "api_documentation.html")
        
        # 检查文件是否存在
        if not os.path.exists(html_file_path):
            return HTMLResponse(
                content="<html><body><h1>API Documentation not found</h1></body></html>",
                status_code=404
            )
        
        # 读取并返回HTML内容
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content, status_code=200)
        
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Error loading API documentation: {str(e)}</h1></body></html>",
            status_code=500
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MeloTTS API Server')
    parser.add_argument('--port', type=int, default=9003, help='Port number to run the server on (default: 9003)')
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
