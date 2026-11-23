
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

RUTA = "modelos/qwen3_4b"

config_4bits = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Cargando Qwen3-4B en 4 bits...")

tokenizer = AutoTokenizer.from_pretrained(
    RUTA,
    local_files_only=True,
)

modelo = AutoModelForCausalLM.from_pretrained(
    RUTA,
    quantization_config=config_4bits,
    device_map="auto",
    dtype=torch.float16,
    local_files_only=True,
)

mensajes = [
    {
        "role": "user",
        "content": (
            "Escribe en español tres oraciones de un cuento "
            "sobre Milo, un gato astronauta que busca a su "
            "mejor amigo en un planeta desconocido."
        ),
    }
]

texto = tokenizer.apply_chat_template(
    mensajes,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

entradas = tokenizer(
    [texto],
    return_tensors="pt",
).to(modelo.device)

with torch.inference_mode():
    salida = modelo.generate(
        **entradas,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.7,
        top_p=0.85,
        repetition_penalty=1.12,
        pad_token_id=tokenizer.eos_token_id,
    )

respuesta = tokenizer.decode(
    salida[0][entradas["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)

print("\nRESPUESTA:\n", respuesta)

if torch.cuda.is_available():
    print(
        "\nPico de VRAM:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB",
    )
