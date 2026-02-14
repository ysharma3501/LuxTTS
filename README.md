# LuxTTS
<p align="center">
  <a href="https://huggingface.co/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
  &nbsp;
  <a href="https://colab.research.google.com/drive/1cDaxtbSDLRmu6tRV_781Of_GSjHSo1Cu?usp=sharing">
    <img src="https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white" alt="Colab Notebook">
  </a>
</p>

LuxTTS is an lightweight zipvoice based text-to-speech model designed for high quality voice cloning and realistic generation at speeds exceeding 150x realtime.

https://github.com/user-attachments/assets/a3b57152-8d97-43ce-bd99-26dc9a145c29

### Alternative Implementations
- Apple Silicon MLX implementation (community-maintained): [jishnuvenugopal/LuxTTS-mlx](https://github.com/jishnuvenugopal/LuxTTS-mlx)
- PyPI package: [LuxTTS-mlx](https://pypi.org/project/LuxTTS-mlx/)

### The main features are
- Voice cloning: SOTA voice cloning on par with models 10x larger.
- Clarity: Clear 48khz speech generation unlike most TTS models which are limited to 24khz.
- Speed: Reaches speeds of 150x realtime on a single GPU and faster then realtime on CPU's as well.
- Efficiency: Fits within 1gb vram meaning it can fit in any local gpu.

## Usage
You can try it locally, colab, or spaces.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cDaxtbSDLRmu6tRV_781Of_GSjHSo1Cu?usp=sharing)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YatharthS/LuxTTS)

#### Simple installation:
```
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS
pip install -r requirements.txt
```

#### Load model:
```python
from zipvoice.luxvoice import LuxTTS

# load model on GPU
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')

# load model on CPU
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=2)

# load model on MPS for macs
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')
```

#### Simple inference
```python
import soundfile as sf
from IPython.display import Audio

text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

## encode audio(takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, rms=0.01)

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech
if display is not None:
  display(Audio(final_wav, rate=48000))
```

#### Inference with sampling params:
```python
import soundfile as sf
from IPython.display import Audio

text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

rms = 0.01 ## higher makes it sound louder(0.01 or so recommended)
t_shift = 0.9 ## sampling param, higher can sound better but worse WER
num_steps = 4 ## sampling param, higher sounds better but takes longer(3-4 is best for efficiency)
speed = 1.0 ## sampling param, controls speed of audio(lower=slower)
return_smooth = False ## sampling param, makes it sound smoother possibly but less cleaner
ref_duration = 5 ## Setting it lower can speedup inference, set to 1000 if you find artifacts.

## encode audio(takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, duration=ref_duration, rms=rms)

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=num_steps, t_shift=t_shift, speed=speed, return_smooth=return_smooth)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech
if display is not None:
  display(Audio(final_wav, rate=48000))
```
## Tips
- Please use at minimum a 3 second audio file for voice cloning.
- You can use return_smooth = True if you hear metallic sounds.
- Lower t_shift for less possible pronunciation errors but worse quality and vice versa.

  
## Info

Q: How is this different from ZipVoice?

A: LuxTTS uses the same architecture but distilled to 4 steps with an improved sampling technique. It also uses a custom 48khz vocoder instead of the default 24khz version.

Q: Can it be even faster?

A: Yes, currently it uses float32. Float16 should be significantly faster(almost 2x).

## Roadmap

- [x] Release model and code
- [x] Huggingface spaces demo
- [x] Release MPS support (thanks to @builtbybasit)
- [ ] Release LuxTTS v1.5
- [ ] Release code for float16 inference

## Acknowledgments

- [ZipVoice](https://github.com/k2-fsa/ZipVoice) for their excellent code and model.
- [Vocos](https://github.com/gemelo-ai/vocos.git) for their great vocoder.
  
## Final Notes

The model and code are licensed under the Apache-2.0 license. See LICENSE for details.

Stars/Likes would be appreciated, thank you.

Email: yatharthsharma350@gmail.com
