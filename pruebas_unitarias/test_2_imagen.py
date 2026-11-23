import os
import time
import torch
from diffusers import AutoPipelineForText2Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "sd_turbo")
CARPETA_SALIDA = os.path.join(BASE_DIR, "pruebas_unitarias", "salida_imagenes")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Cargando SD-Turbo en dispositivo: {device}...")

pipe = AutoPipelineForText2Image.from_pretrained(
    RUTA_MODELO,
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None
).to(device)

# Estrategia de consistencia visual: Estilo fijo + Personaje fijo + Semilla fija
ESTILO_FIJO = "children storybook digital illustration, vibrant colors, soft lighting, detailed art style"
PERSONAJE_FIJO = "a cute white cat astronaut wearing a small white space suit and glass helmet"

prompts_escenas = [
    "standing on a planet covered in glowing blue crystals",
    "examining giant floating crystal formations with curiosity"
]

for i, accion in enumerate(prompts_escenas, start=1):
    prompt_completo = f"{ESTILO_FIJO}, {PERSONAJE_FIJO}, {accion}"
    print(f"\nGenerando imagen {i}...")
    inicio = time.time()

    # Semilla fija (42) para mantener coherencia entre escenas
    generador = torch.Generator(device=device).manual_seed(42)
    imagen = pipe(
        prompt=prompt_completo,
        num_inference_steps=2,   # SD-Turbo usa entre 1 y 4 pasos
        guidance_scale=0.0,      # Obligatorio en 0.0 para modelos Turbo
        generator=generador
    ).images[0]

    ruta_guardado = os.path.join(CARPETA_SALIDA, f"escena_{i}.png")
    imagen.save(ruta_guardado)
    print(f"[OK] Guardada en {ruta_guardado} (Tiempo: {time.time() - inicio:.2f}s)")