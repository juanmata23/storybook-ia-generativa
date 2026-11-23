import os
import time
import wave
from piper.voice import PiperVoice

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_VOZ = os.path.join(BASE_DIR, "modelos", "piper_voz", "es", "es_MX", "claude", "high", "es_MX-claude-high.onnx")
CARPETA_SALIDA = os.path.join(BASE_DIR, "pruebas_unitarias", "salida_audio")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

print("Cargando modelo de voz Piper TTS...")
voz = PiperVoice.load(RUTA_VOZ)

texto_escena = (
    "Luna, un gato astronauta, encuentra un planeta en el espacio, "
    "cubierto de cristales brillantes que resplandecen bajo las estrellas."
)

ruta_wav = os.path.join(CARPETA_SALIDA, "escena_1_voz.wav")
inicio = time.time()

with wave.open(ruta_wav, "wb") as archivo_wav:
    voz.synthesize_wav(texto_escena, archivo_wav)

print(f"[OK] Audio de voz generado en: {ruta_wav} (Tiempo: {time.time() - inicio:.2f}s)")