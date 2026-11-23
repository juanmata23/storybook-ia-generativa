import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ruta relativa automática: busca la carpeta 'modelos/qwen3_texto' un nivel arriba
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "qwen3_texto")

print(f"Buscando modelo en: {RUTA_MODELO}")
if not os.path.exists(RUTA_MODELO):
    raise FileNotFoundError("No se encontró la carpeta del modelo. Verifica dónde se descargó.")

print("Cargando tokenizador y modelo local...")
tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO)
model = AutoModelForCausalLM.from_pretrained(
    RUTA_MODELO,
    torch_dtype="auto",
    device_map="auto"
)

def generar_cuento_prueba(idea: str, genero: str):
    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres un generador de cuentos estructurados. Responde ÚNICAMENTE con un objeto JSON válido, "
                "sin texto adicional ni bloques markdown. El JSON debe tener exactamente 5 escenas."
            )
        },
        {
            "role": "user",
            "content": f"""Crea un cuento de género '{genero}' basado en la idea: '{idea}'.
Mantén el mismo personaje principal y estilo en las 5 escenas (inicio, desarrollo y desenlace).
Usa exactamente esta estructura JSON:
{{
  "titulo": "Título del cuento",
  "personaje": "Descripción física fija del protagonista en inglés",
  "escenas": [
    {{
      "numero": 1,
      "texto": "Narración corta en español (2 o 3 oraciones).",
      "prompt_imagen": "Visual scene description in English including the character",
      "prompt_sonido": "Ambient sound effect in English (e.g. rain, forest wind, footsteps)"
    }}
  ]
}}"""
        }
    ]

    texto_entrada = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([texto_entrada], return_tensors="pt").to(model.device)

    print("Generando historia de 5 escenas...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.7,
            do_sample=True
        )

    tokens_generados = outputs[0][len(inputs.input_ids[0]):]
    respuesta = tokenizer.decode(tokens_generados, skip_special_tokens=True).strip()

    if "</think>" in respuesta:
        respuesta = respuesta.split("</think>")[-1].strip()
    if respuesta.startswith("```json"):
        respuesta = respuesta.replace("```json", "").replace("```", "").strip()

    print("\n--- RESULTADO CRUDO ---")
    print(respuesta)

    datos = json.loads(respuesta)
    print(f"\n[OK] Título: {datos['titulo']}")
    print(f"[OK] Escenas generadas: {len(datos['escenas'])}")

if __name__ == "__main__":
    generar_cuento_prueba("Un gato astronauta descubre un planeta hecho de cristal", "Ciencia ficción")