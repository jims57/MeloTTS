# 作者: Jimmy Gan
# 日期: 2024-01-25
# 功能: 将所有PCM音频块合并为单个WAV文件

import os
import wave
import glob
import re

def combine_pcm_to_wav(pcm_chunks_dir, output_wav_path, sample_rate=8000, channels=1, sample_width=2):
    """
    将所有PCM文件按顺序合并为单个WAV文件
    
    参数:
        pcm_chunks_dir: 包含PCM块文件的目录
        output_wav_path: 输出合并WAV文件的路径
        sample_rate: 采样率，单位Hz（默认：8000）
        channels: 声道数（默认：1，单声道）
        sample_width: 采样宽度，单位字节（默认：2，16位）
    """
    # 获取所有PCM文件并按文件名排序以保持顺序
    pcm_files = glob.glob(os.path.join(pcm_chunks_dir, "*.pcm"))
    # 通过提取块编号进行数字排序
    pcm_files.sort(key=lambda x: int(re.search(r'chunk_(\d+)\.pcm', os.path.basename(x)).group(1)))
    
    combined_pcm_data = b''
    
    # 按顺序读取并合并所有PCM文件
    for pcm_file in pcm_files:
        print(f"正在读取 {os.path.basename(pcm_file)}")
        with open(pcm_file, 'rb') as f:
            pcm_data = f.read()
            combined_pcm_data += pcm_data
    
    # 创建合并的WAV文件
    with wave.open(output_wav_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(combined_pcm_data)

def main():
    # 定义目录路径
    pcm_chunks_dir = "pcm_chunks"
    combined_wav_dir = "combined_wav"
    
    # 如果输出目录不存在则创建
    os.makedirs(combined_wav_dir, exist_ok=True)
    
    # 定义输出文件路径
    output_wav_path = os.path.join(combined_wav_dir, "combined_audio.wav")
    
    print("正在将所有PCM块合并为单个WAV文件...")
    combine_pcm_to_wav(pcm_chunks_dir, output_wav_path, sample_rate=8000, channels=1)
    print(f"合并的WAV文件已保存: {output_wav_path}")

if __name__ == "__main__":
    main()
