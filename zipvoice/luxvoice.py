import torch
from zipvoice.modeling_utils import process_audio, generate, load_models_gpu, load_models_cpu
from zipvoice.onnx_modeling import generate_cpu

class LuxTTS:
    """
    LuxTTS class for encoding prompt and generating speech on cpu/cuda/mps.
    """

    def __init__(self, model_path='YatharthS/LuxTTS', device='cuda', threads=4):
        if model_path == 'YatharthS/LuxTTS':
            model_path = None

        # Auto-detect better device if cuda is requested but not available
        if device == 'cuda' and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("CUDA not available, switching to MPS")
                device = 'mps'
            else:
                print("CUDA not available, switching to CPU")
                device = 'cpu'

        if device == 'cpu':
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_cpu(model_path, threads)
            print("Loading model on CPU")
        else:
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_gpu(model_path, device=device)
            print("Loading model on GPU")

        self.model = model
        self.feature_extractor = feature_extractor
        self.vocos = vocos
        self.tokenizer = tokenizer
        self.transcriber = transcriber
        self.device = device
        self.vocos.freq_range = 12000



    def encode_prompt(self, prompt_audio, duration=5, rms=0.001, text=None):
        """encodes audio prompt according to duration and rms(volume control)"""
        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = process_audio(prompt_audio, self.transcriber, self.tokenizer, self.feature_extractor, self.device, target_rms=rms, duration=duration, text=text)
        encode_dict = {"prompt_tokens": prompt_tokens, 'prompt_features_lens': prompt_features_lens, 'prompt_features': prompt_features, 'prompt_rms': prompt_rms}

        return encode_dict

    def generate_speech(self, text, encode_dict, num_steps=4, guidance_scale=3.0, t_shift=0.5, speed=1.0, return_smooth=False):
        """encodes text and generates speech using flow matching model according to steps, guidance scale, and t_shift(like temp)"""

        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = encode_dict.values()

        if return_smooth == True:
            self.vocos.return_48k = False
        else:
            self.vocos.return_48k = True

        if self.device == 'cpu':
            final_wav = generate_cpu(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, self.model, self.vocos, self.tokenizer, num_step=num_steps, guidance_scale=guidance_scale, t_shift=t_shift, speed=speed)
        else:
            final_wav = generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, self.model, self.vocos, self.tokenizer, num_step=num_steps, guidance_scale=guidance_scale, t_shift=t_shift, speed=speed)

        return final_wav.cpu()
