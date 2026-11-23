import os
import time
import torch
import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "musicgen_small")
CARPETA_SALIDA = os.path.join(BASE_DIR, "pruebas_unitarias", "salida_audio")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cargando MusicGen Small en: {device}...")

processor = AutoProcessor.from_pretrained(RUTA_MODELO)
model = MusicgenForConditionalGeneration.from_pretrained(RUTA_MODELO).to(device)

prompt_musica = ["mysterious ambient sci-fi synth, gentle crystal chimes, calm space exploration background"]
print(f"Generando ambientación musical: '{prompt_musica[0]}'...")
inicio = time.time()

inputs = processor(
    text=prompt_musica,
    padding=True,
    return_tensors="pt"
).to(device)

# max_new_tokens=256 genera ~5 segundos de audio (50 tokens ≈ 1 segundo)
with torch.no_grad():
    audio_values = model.generate(**inputs, max_new_tokens=256)

sampling_rate = model.config.audio_encoder.sampling_rate
audio_numpy = audio_values[0, 0].cpu().numpy()

ruta_wav = os.path.join(CARPETA_SALIDA, "escena_1_musicgen.wav")
scipy.io.wavfile.write(ruta_wav, rate=sampling_rate, data=audio_numpy)
print(f"[OK] Música de ambiente guardada en: {ruta_wav} (Tiempo: {time.time() - inicio:.2f}s)")