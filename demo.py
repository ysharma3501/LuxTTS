import gradio as gr
import numpy as np
import soundfile as sf
import tempfile
import os
from zipvoice.luxvoice import LuxTTS

# Initialize LuxTTS model (only once at startup)
print("Loading LuxTTS model...")
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda', threads=2)  # Use device='cpu' for CPU
print("Model loading complete")

def generate(audio, text, rms, num_steps, t_shift):
    if audio is None:
        return None, "Error: Please upload an audio file"
    
    if not text or text.strip() == "":
        return None, "Error: Please enter text"
    
    try:
        sample_rate, audio_data = audio
        
        print(f"Received text: {text}")
        print(f"Sample rate: {sample_rate}")
        print(f"Audio data shape: {audio_data.shape}")
        print(f"Parameters - RMS: {rms}, Num steps: {num_steps}, T-shift: {t_shift}")
        
        # Save audio to temporary file (LuxTTS requires file path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_data, sample_rate)
        
        # Encode audio (first run may take ~10 seconds)
        print("Encoding audio...")
        encoded_prompt = lux_tts.encode_prompt(tmp_path, rms=rms)
        
        # Generate speech
        print("Generating speech...")
        final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=num_steps, t_shift=t_shift)
        
        # Convert to numpy array
        final_wav = final_wav.numpy().squeeze()
        
        # Delete temporary file
        os.unlink(tmp_path)
        
        print("Speech generation complete")
        
        # Return with 48000Hz sample rate
        return (48000, final_wav), "✓ Speech generation complete"
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LuxTTS Voice Cloning")
    gr.Markdown("Upload a reference audio and enter text to generate speech")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Reference Audio (WAV/MP3)",
                type="numpy",
                sources=["upload", "microphone"]
            )
            text_input = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter the text you want to generate as speech",
                lines=3
            )
            
            # Parameter controls
            gr.Markdown("### Generation Parameters")
            rms_slider = gr.Slider(
                minimum=0.001,
                maximum=0.1,
                value=0.01,
                step=0.001,
                label="RMS (Volume normalization)",
                info="Root Mean Square for audio normalization (default: 0.01)"
            )
            num_steps_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=4,
                step=1,
                label="Number of Steps",
                info="Inference steps - higher values may improve quality but take longer (default: 4)"
            )
            t_shift_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="T-Shift",
                info="Temperature shift parameter (default: 0.9)"
            )
            
            submit_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(
                label="Generated Speech (WAV)",
                type="numpy"
            )
            status_output = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    # Handle button click
    submit_btn.click(
        fn=generate,
        inputs=[audio_input, text_input, rms_slider, num_steps_slider, t_shift_slider],
        outputs=[audio_output, status_output]
    )
    
    # Usage instructions
    gr.Markdown("""
    ## How to Use
    1. **Reference Audio**: Upload a WAV or MP3 file of the voice you want to clone, or record from microphone
    2. **Text Input**: Enter the text you want to generate as speech (English recommended)
    3. **Adjust Parameters** (optional):
       - **RMS**: Controls volume normalization (0.001-0.1, default: 0.01)
       - **Number of Steps**: Inference quality/speed tradeoff (1-20, default: 4)
       - **T-Shift**: Temperature parameter affecting generation (0.0-1.0, default: 0.9)
    4. Click the **Generate Speech** button
    5. Once processing is complete, the generated audio will appear on the right
    
    **Notes**: 
    - First run may take ~10 seconds for audio encoding initialization
    - Clear reference audio with minimal background noise is recommended
    - Use `device='cuda'` for GPU or `device='cpu'` for CPU in the code
    - Higher num_steps values may improve quality but increase processing time
    """)

if __name__ == "__main__":
    demo.launch()
