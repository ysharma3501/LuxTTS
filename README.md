Project quick start :https://github.com/ysharma3501/LuxTTS

update note:

2026-01-29
1. Skip Whisper recognition if text exists in speaker.yml; 
2. Add 50ms/80ms silence via NumPy in post-processing; 
3. Language-specific t_shift/guidance_scale by text proportion; 
4. Independent Chinese speech rate with token padding coefficient.
