#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lux-TTS 服务端实现
支持HTTP接口和SSE接口
参数包括speaker、text
"""

import os
import yaml
import base64
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import time
from typing import Generator, Dict, Any, Tuple, List
from contextlib import asynccontextmanager
import soundfile as sf
from io import BytesIO
import logging
from zipvoice.luxvoice import LuxTTS
import asyncio
from dataclasses import dataclass
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 配置参数区域 ====================
# 服务配置
NUM_MODEL_INSTANCES = int(os.getenv('NUM_MODEL_INSTANCES', '24'))  # 默认24个实例（适合40GB显存）
MODEL_ACQUIRE_TIMEOUT = int(os.getenv('MODEL_ACQUIRE_TIMEOUT', '10'))  # 获取模型超时时间（秒）

# 线程池配置
import concurrent.futures
CPU_CORES = 24  # vCPU核心数
MAX_WORKERS = min(int(CPU_CORES * 1.2), 36)  # 线程池大小
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# 音频配置
ORIGINAL_SAMPLE_RATE = 48000  # 原始采样率
TARGET_SAMPLE_RATE = 16000    # 目标采样率

# 模型参数
rms = 0.01  ## higher makes it sound louder(0.01 or so recommended)
t_shift = 0.65  ## sampling param, higher can sound better but worse WER
num_steps = 4  ## sampling param, higher sounds better but takes longer(3-4 is best for efficiency)
return_smooth = True  ## sampling param, makes it sound smoother possibly but less cleaner

# 静音配置
PRE_SILENCE_MS = 50  # 前静音（毫秒）
POST_SILENCE_MS = 80  # 后静音（毫秒）

# 语言相关配置
CHINESE_T_SHIFT = 0.65  # 中文t_shift值
CHINESE_GUIDANCE_SCALE = 3.0  # 中文guidance_scale值
ENGLISH_T_SHIFT = 0.85  # 英文t_shift值
ENGLISH_GUIDANCE_SCALE = 3.0  # 英文guidance_scale值
CHINESE_SPEED_ADJUSTMENT = 0.8  # 中文速度调整系数
# =====================================================

# 存储发音人特征数据
speaker_embeddings = {}

# 全局重采样器（在启动时初始化）
gpu_resampler = None
cpu_resampler = None


class TTSRequest(BaseModel):
    """TTS请求体模型"""
    text: str
    speaker: str
    speed: float = 1.0  # 语速控制参数，默认1.0，范围建议0.5-2.0


class SpeakerInfo:
    """发音人信息类"""

    def __init__(self, key: str, wav_path: str, text: str = None):
        self.key = key
        self.wav_path = wav_path
        self.text = text
        self.embedding = None


@dataclass
class ModelInstanceStats:
    """模型实例统计信息"""
    instance_id: int
    total_requests: int = 0
    total_inference_time: float = 0.0
    last_used_time: datetime = None
    is_busy: bool = False


class ModelInstance:
    """单个模型实例封装"""
    
    def __init__(self, instance_id: int, model: Any):
        self.instance_id = instance_id
        self.model = model
        self.stats = ModelInstanceStats(instance_id=instance_id)
        # 不再需要锁，由队列管理并发
    
    def update_stats(self, inference_time: float):
        """更新统计信息"""
        self.stats.total_requests += 1
        self.stats.total_inference_time += inference_time
        self.stats.last_used_time = datetime.now()
        self.stats.is_busy = False


class ModelPool:
    """模型实例池管理器（使用 asyncio.Queue 实现真正的异步）"""
    
    def __init__(self, num_instances: int):
        self.num_instances = num_instances
        self.instances: List[ModelInstance] = []
        self.queue: asyncio.Queue = None  # 在 lifespan 中初始化
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self, model_path: str, device: str = 'cuda', threads: int = 4):
        """初始化所有模型实例"""
        async with self._lock:
            if self._initialized:
                logger.warning("ModelPool already initialized")
                return
            
            # 创建异步队列
            self.queue = asyncio.Queue(maxsize=self.num_instances)
            
            logger.info(f"Initializing {self.num_instances} model instances...")
            logger.info(f"Using device: {device}, threads per instance: {threads}")
            
            for i in range(self.num_instances):
                try:
                    logger.info(f"Loading model instance {i+1}/{self.num_instances}...")
                    model = LuxTTS(model_path, device=device, threads=threads)
                    instance = ModelInstance(instance_id=i, model=model)
                    self.instances.append(instance)
                    await self.queue.put(instance)  # 使用 await
                    logger.info(f"Model instance {i} loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load model instance {i}: {e}")
                    raise
            
            self._initialized = True
            logger.info(f"Successfully initialized {len(self.instances)} model instances")
    
    async def acquire(self, timeout: int = MODEL_ACQUIRE_TIMEOUT) -> ModelInstance:
        """获取一个空闲的模型实例（真正的异步等待）"""
        if not self._initialized:
            raise RuntimeError("ModelPool not initialized")
        
        try:
            # 使用 asyncio.wait_for 实现超时控制
            instance = await asyncio.wait_for(
                self.queue.get(),
                timeout=timeout
            )
            instance.stats.is_busy = True
            logger.debug(f"Acquired model instance {instance.instance_id}")
            return instance
        except asyncio.TimeoutError:
            raise TimeoutError(f"Failed to acquire model instance within {timeout} seconds. All instances are busy.")
    
    def release(self, instance: ModelInstance):
        """释放模型实例回池中（同步方法，立即返回）"""
        instance.stats.is_busy = False
        # 使用 put_nowait 立即放回，不会阻塞
        self.queue.put_nowait(instance)
        logger.debug(f"Released model instance {instance.instance_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取实例池统计信息"""
        total_requests = sum(inst.stats.total_requests for inst in self.instances)
        total_inference_time = sum(inst.stats.total_inference_time for inst in self.instances)
        busy_count = sum(1 for inst in self.instances if inst.stats.is_busy)
        
        instance_details = []
        for inst in self.instances:
            avg_time = (inst.stats.total_inference_time / inst.stats.total_requests 
                       if inst.stats.total_requests > 0 else 0)
            instance_details.append({
                "instance_id": inst.instance_id,
                "total_requests": inst.stats.total_requests,
                "total_inference_time": round(inst.stats.total_inference_time, 2),
                "avg_inference_time": round(avg_time, 3),
                "last_used": inst.stats.last_used_time.isoformat() if inst.stats.last_used_time else None,
                "is_busy": inst.stats.is_busy
            })
        
        return {
            "total_instances": self.num_instances,
            "busy_instances": busy_count,
            "available_instances": self.num_instances - busy_count,
            "total_requests": total_requests,
            "total_inference_time": round(total_inference_time, 2),
            "avg_inference_time": round(total_inference_time / total_requests, 3) if total_requests > 0 else 0,
            "instances": instance_details
        }


def load_speakers_config(config_path: str) -> Dict[str, SpeakerInfo]:
    """加载发音人配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    speakers_info = {}
    for speaker_data in config['speaker']:
        key = speaker_data['key']
        wav_path = os.path.join(os.path.dirname(config_path), speaker_data['wav'])
        # 读取配置文件中的参考文本
        text = speaker_data.get('text', None)

        # 创建SpeakerInfo对象
        speaker_info = SpeakerInfo(key, wav_path, text)
        speakers_info[key] = speaker_info

        logger.info(f"Loaded speaker: {key}, wav: {wav_path}, text: {text}")

    return speakers_info


def load_speaker_embeddings(model: Any, speakers_info: Dict[str, SpeakerInfo]):
    """加载发音人特征嵌入"""
    global speaker_embeddings

    for key, speaker_info in speakers_info.items():
        try:
            # 使用模型提取发音人特征
            # 从参考音频和文本生成特征
            data, sample_rate = sf.read(speaker_info.wav_path)
            # 计算时长（以秒为单位）
            ref_duration = len(data) / float(sample_rate)
            # 传递配置文件中指定的参考文本
            ref_embedding = model.encode_prompt(speaker_info.wav_path, duration=ref_duration, rms=rms, text=speaker_info.text)
            speaker_embeddings[key] = ref_embedding
            logger.info(f"Generated embedding for speaker: {key},duration: {ref_duration}, text: {speaker_info.text}")

        except Exception as e:
            logger.error(f"Error loading embedding for speaker {key}: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global speaker_embeddings

    logger.info(f"Initializing Lux-TTS with {NUM_MODEL_INSTANCES} model instances...")

    try:
        # 加载发音人配置
        config_path = os.path.join(os.path.dirname(__file__), 'audios', 'speaker.yml')
        speakers_info = load_speakers_config(config_path)

        # 初始化全局重采样器
        logger.info("Initializing global audio resamplers...")
        initialize_resamplers()

        # 初始化模型实例池
        model_path = '/opt/LuxTTS/models'
        model_pool = ModelPool(num_instances=NUM_MODEL_INSTANCES)
        
        try:
            # 初始化所有模型实例
            await model_pool.initialize(model_path, device='cuda', threads=2)
            
            # 使用第一个实例加载发音人特征（所有实例共享）
            first_instance = await model_pool.acquire()
            try:
                load_speaker_embeddings(first_instance.model, speakers_info)
            finally:
                model_pool.release(first_instance)
            
        except Exception as e:
            logger.error(f"Failed to initialize model pool: {e}")
            raise

        # 将模型池和配置存储在应用状态中
        app.state.model_pool = model_pool
        app.state.speakers_info = speakers_info

        logger.info(f"Successfully initialized {NUM_MODEL_INSTANCES} model instances and loaded speaker configurations")

    except Exception as e:
        logger.error(f"Failed to initialize Lux-TTS: {str(e)}")
        raise

    yield
    
    # 清理工作
    logger.info("Shutting down model pool...")


app = FastAPI(
    title="Lux-TTS Server",
    description="Lux-TTS API Service",
    lifespan=lifespan
)


def initialize_resamplers():
    """初始化全局重采样器"""
    global gpu_resampler, cpu_resampler
    
    try:
        # 创建 GPU 重采样器
        gpu_resampler = torchaudio.transforms.Resample(
            orig_freq=ORIGINAL_SAMPLE_RATE,
            new_freq=TARGET_SAMPLE_RATE,
            dtype=torch.float32
        ).to('cuda')
        logger.info(f"GPU resampler initialized: {ORIGINAL_SAMPLE_RATE}Hz -> {TARGET_SAMPLE_RATE}Hz")
    except Exception as e:
        logger.warning(f"Failed to initialize GPU resampler: {e}")
        gpu_resampler = None
    
    # 创建 CPU 重采样器（作为后备）
    cpu_resampler = torchaudio.transforms.Resample(
        orig_freq=ORIGINAL_SAMPLE_RATE,
        new_freq=TARGET_SAMPLE_RATE,
        dtype=torch.float32
    )
    logger.info("CPU resampler initialized as fallback")


def resample_audio_gpu(audio_tensor: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """使用预初始化的GPU重采样器进行音频重采样
    
    Args:
        audio_tensor: 输入音频张量 (channels, samples) 或 (samples,)
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        重采样后的音频张量
    """
    global gpu_resampler, cpu_resampler
    
    try:
        # 确保数据类型为 float32
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()
        
        # 确保音频在 GPU 上
        if not audio_tensor.is_cuda and device == 'cuda':
            audio_tensor = audio_tensor.to(device)
        
        # 确保是 2D 张量 (channels, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # 使用预初始化的 GPU 重采样器
        if gpu_resampler is not None:
            resampled = gpu_resampler(audio_tensor)
            return resampled
        else:
            raise RuntimeError("GPU resampler not initialized")
    
    except Exception as e:
        logger.warning(f"GPU resampling failed: {e}, falling back to CPU")
        # 降级到 CPU 重采样
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()
        
        if cpu_resampler is not None:
            return cpu_resampler(audio_tensor)
        else:
            # 最后的兜底方案
            fallback_resampler = torchaudio.transforms.Resample(
                orig_freq=ORIGINAL_SAMPLE_RATE,
                new_freq=TARGET_SAMPLE_RATE,
                dtype=torch.float32
            )
            return fallback_resampler(audio_tensor)


def detect_language(text: str) -> str:
    """检测文本语言类型"""
    # 统计中文字符数量
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    # 如果中文字符占比超过30%，认为是中文
    if chinese_chars / len(text) > 0.3:
        return 'chinese'
    else:
        return 'english'

def synthesize_speech(text: str, speaker: str, model: Any, speed: float = 1.0) -> Tuple[torch.Tensor, float, float]:
    """语音合成（仅模型推理，不包含后处理）
    
    Returns:
        Tuple[torch.Tensor, float, float]: (final_wav, inference_time, audio_duration)
    """
    try:
        # 检查发音人是否存在
        if speaker not in app.state.speakers_info:
            raise ValueError(f"Speaker '{speaker}' not found in configuration")

        # 检测语言类型并设置相应参数
        language = detect_language(text)
        if language == 'chinese':
            current_t_shift = CHINESE_T_SHIFT
            current_guidance_scale = CHINESE_GUIDANCE_SCALE
            # 中文需要调整速度，因为EmiliaTokenizer对中文的token化产生较少token
            # 降低速度以补偿token数量较少的问题
            adjusted_speed = speed * CHINESE_SPEED_ADJUSTMENT
        else:
            current_t_shift = ENGLISH_T_SHIFT
            current_guidance_scale = ENGLISH_GUIDANCE_SCALE
            # 英文保持原始速度
            adjusted_speed = speed

        start_time = time.time()
        # 使用预生成的发音人特征进行语音合成
        final_wav = model.generate_speech(text, speaker_embeddings[speaker], num_steps=num_steps, t_shift=current_t_shift,
                                          guidance_scale=current_guidance_scale, speed=adjusted_speed, return_smooth=return_smooth)
        inference_time = time.time() - start_time
        
        # 估算音频时长（基于原始采样率）
        audio_duration = final_wav.shape[-1] / ORIGINAL_SAMPLE_RATE
        
        # 返回 GPU 上的 tensor，推理时间和音频时长
        return final_wav, inference_time, audio_duration

    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise


def post_process_audio(final_wav: torch.Tensor, inference_time: float, audio_duration: float) -> Tuple[bytes, float]:
    """音频后处理（重采样、转换，不占用模型）
    
    Args:
        final_wav: GPU 上的音频 tensor
        inference_time: 推理时间
        audio_duration: 音频时长
    
    Returns:
        Tuple[bytes, float]: (pcm_bytes, rtf)
    """
    try:
        # GPU 重采样：48kHz -> 16kHz
        resampled_wav = resample_audio_gpu(final_wav, device='cuda')
        
        # 转换为 numpy 数组
        resampled_wav = resampled_wav.cpu().numpy().squeeze()
        
        # 添加前后静音
        sample_rate = TARGET_SAMPLE_RATE
        pre_silence_ms = PRE_SILENCE_MS
        post_silence_ms = POST_SILENCE_MS
        
        # 计算静音的样本数
        pre_silence_samples = int(sample_rate * pre_silence_ms / 1000)
        post_silence_samples = int(sample_rate * post_silence_ms / 1000)
        
        # 创建静音数组并拼接
        if pre_silence_samples > 0 or post_silence_samples > 0:
            pre_silence = np.zeros(pre_silence_samples, dtype=np.float32)
            post_silence = np.zeros(post_silence_samples, dtype=np.float32)
            resampled_wav = np.concatenate([pre_silence, resampled_wav, post_silence])
        
        # 重新计算采样后的音频时长
        resampled_duration = len(resampled_wav) / TARGET_SAMPLE_RATE
        rtf = inference_time / resampled_duration if resampled_duration > 0 else 0

        # 转换为 PCM 格式（16-bit signed integer）
        resampled_wav = np.clip(resampled_wav, -1.0, 1.0)
        pcm_data = (resampled_wav * 32767.0).astype(np.int16)
        
        # 直接返回 PCM 字节流（无 WAV 头）
        pcm_bytes = pcm_data.tobytes()
        
        return pcm_bytes, rtf

    except Exception as e:
        logger.error(f"Error in post-processing audio: {str(e)}")
        raise


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Welcome to Lux-TTS Server",
        "endpoints": {
            "http": "/tts/http"
        }
    }


@app.post("/tts/http")
async def tts_http(request: TTSRequest):
    """HTTP接口：接收text和speaker参数，返回base64编码的音频数据"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text parameter is required and cannot be empty")

        if not request.speaker.strip():
            raise HTTPException(status_code=400, detail="Speaker parameter is required and cannot be empty")

        # 阶段 1：获取模型实例并进行推理
        model_instance = None
        final_wav = None
        inference_time = 0
        audio_duration = 0
        instance_id = -1
        
        try:
            model_instance = await app.state.model_pool.acquire()
            instance_id = model_instance.instance_id
            
            logger.debug(f"Instance {instance_id} acquired for inference")
            
            # 在线程池中执行模型推理（仅推理）
            loop = asyncio.get_event_loop()
            final_wav, inference_time, audio_duration = await loop.run_in_executor(
                executor,  # 使用自定义线程池
                synthesize_speech,
                request.text,
                request.speaker,
                model_instance.model,
                request.speed
            )
            
            # 更新实例统计信息
            model_instance.update_stats(inference_time)
            
            logger.debug(f"Instance {instance_id} inference completed, releasing...")
            
        except TimeoutError as te:
            logger.error(f"Timeout acquiring model instance: {str(te)}")
            raise HTTPException(status_code=503, detail="Server is busy. Please try again later.")
        
        finally:
            # 关键：推理完成后立即释放模型实例
            if model_instance is not None:
                app.state.model_pool.release(model_instance)
                logger.debug(f"Instance {instance_id} released")
        
        # 阶段 2：后处理（重采样、转换、编码），不占用模型
        if final_wav is None:
            raise HTTPException(status_code=500, detail="TTS synthesis failed")
        
        # 在线程池中执行后处理
        loop = asyncio.get_event_loop()
        audio_bytes, rtf = await loop.run_in_executor(
            executor,  # 使用自定义线程池
            post_process_audio,
            final_wav,
            inference_time,
            audio_duration
        )
        
        # 编码为base64
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

        return JSONResponse(content={
            "audio": base64_audio,
            "format": "pcm",
            "sample_rate": TARGET_SAMPLE_RATE,
            "bit_depth": 16,
            "channels": 1,
            "speaker": request.speaker,
            "text": request.text,
            "speed": request.speed,
            "rtf": rtf,
            "instance_id": instance_id
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in HTTP TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


@app.get("/speakers")
async def get_speakers():
    """获取可用的发音人列表"""
    if hasattr(app.state, 'speakers_info'):
        speakers = list(app.state.speakers_info.keys())
    else:
        speakers = []

    return {"speakers": speakers}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        if not hasattr(app.state, 'model_pool'):
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "Model pool not initialized"}
            )
        
        stats = app.state.model_pool.get_stats()
        
        return {
            "status": "healthy",
            "model_instances": stats["total_instances"],
            "available_instances": stats["available_instances"],
            "busy_instances": stats["busy_instances"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )


@app.get("/stats")
async def get_statistics():
    """获取详细的实例池统计信息"""
    try:
        if not hasattr(app.state, 'model_pool'):
            raise HTTPException(status_code=503, detail="Model pool not initialized")
        
        stats = app.state.model_pool.get_stats()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os
    
    # 从环境变量获取 workers 数量，默认 1
    workers = int(os.getenv('UVICORN_WORKERS', '1'))
    
    # 多 worker 模式仅适用于 Linux/Unix，Windows 不支持
    if workers > 1 and os.name == 'nt':
        logger.warning("Multi-worker mode is not supported on Windows, falling back to single worker")
        workers = 1
    
    logger.info(f"Starting server with {workers} worker(s)...")
    
    # 多 worker 模式需要使用导入字符串
    if workers > 1:
        uvicorn.run(
            "lux_server:app",  # 使用导入字符串
            host="0.0.0.0",
            port=7000,
            workers=workers,
            log_level="info"
        )
    else:
        # 单 worker 模式可以直接使用 app 对象
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=7000,
            log_level="info"
        )

