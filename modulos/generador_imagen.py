"""
STORYBOOK - GENERADOR DE IMÁGENES CON SD-TURBO

Qué cambió respecto a la versión anterior:
- Escenario POR ESCENA (campo `lugar_en`), no uno global para las cinco láminas.
- Plano de cámara distinto en cada escena (planos cíclicos) para variar la composición.
- Semilla distinta por escena (antes: la misma en las cinco, lo que repetía la imagen).
- Prompt con menos tokens fijos (estilo corto + personaje corto) para dejar espacio a la escena.
- Tres estrategias de consistencia comparables (STORYBOOK_IMG_MODO):
    "semilla_fija"   -> comportamiento anterior (misma semilla; solo para comparar en el informe)
    "semilla_escena" -> descripción fija del personaje + semilla distinta por escena (por defecto)
    "img2img"        -> escenas 2-5 parten de la escena 1 (más parecido, menos variedad)
- Registro de prompts, semillas y modo en prompts_usados.json (evidencia para el informe).
"""

import gc
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata

import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from PIL import Image


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "sd_turbo")

NUM_ESCENAS = 5
RESOLUCION = 512
PASOS_INFERENCIA = 4
GUIDANCE_SCALE = 0.0
MAX_TOKENS_PROMPT = 75
USAR_CACHE = True

MODO_CONSISTENCIA = os.environ.get("STORYBOOK_IMG_MODO", "semilla_escena")
MODOS_VALIDOS = ("semilla_fija", "semilla_escena", "img2img")
STRENGTH_IMG2IMG = float(os.environ.get("STORYBOOK_IMG_STRENGTH", "0.75"))  # pasos*strength >= 1

# Un plano distinto por escena: varía la composición y evita imágenes repetidas.
PLANOS = [
    "wide establishing shot",
    "medium shot",
    "dynamic low angle shot",
    "close-up shot",
    "wide shot",
]


# ============================================================
# ESTILOS NORMALIZADOS POR GÉNERO (cortos: dejan tokens para la escena)
# ============================================================

ESTILOS_GENERO = {
    "infantil": "children's storybook watercolor illustration, soft warm light",
    "fantasia": "fantasy storybook painting, painterly, enchanted light",
    "ciencia ficcion": "sci-fi storybook illustration, cinematic futuristic light",
    "terror": "dark gothic storybook illustration, eerie moonlight, mist",
    "misterio": "mysterious noir storybook painting, lantern light, deep shadows",
    "aventura": "classic adventure storybook illustration, vibrant colors",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_clave(texto: str) -> str:
    """Normaliza tildes y caracteres corruptos de consola (ej. 'Ciencia Ficci?n')."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z\s]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def resolver_estilo(genero: str) -> tuple:
    clave = normalizar_clave(genero)
    if "ciencia" in clave or "ficc" in clave:
        return "ciencia ficcion", ESTILOS_GENERO["ciencia ficcion"]
    if "terror" in clave or "horror" in clave:
        return "terror", ESTILOS_GENERO["terror"]
    if "misterio" in clave:
        return "misterio", ESTILOS_GENERO["misterio"]
    if "aventura" in clave:
        return "aventura", ESTILOS_GENERO["aventura"]
    if "infantil" in clave:
        return "infantil", ESTILOS_GENERO["infantil"]
    return "fantasia", ESTILOS_GENERO["fantasia"]


def limpiar_prompt(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip(" ,.;")


def crear_semilla(texto: str) -> int:
    digest = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**32)


def semilla_para_escena(semilla_base: int, indice: int) -> int:
    if MODO_CONSISTENCIA == "semilla_fija":
        return semilla_base
    return (semilla_base + indice * 1009) % (2**32)


def contar_tokens_clip(tokenizer, texto: str) -> int:
    return len(tokenizer(texto, add_special_tokens=True, truncation=False)["input_ids"])


def extraer_nombre_propio(personaje_raw: str) -> str:
    match = re.match(r"^([A-Z][a-z]{2,15})\s*,\s*(.+)$", personaje_raw)
    return match.group(1) if match else ""


def obtener_personaje(datos_historia: dict) -> tuple:
    """(descripcion_visual_fija, nombre_propio). Sin nombre propio y con sujeto único."""
    raw = limpiar_prompt(datos_historia.get("personaje_en", ""))
    if not raw:
        return "a single storybook protagonist", ""

    nombre = extraer_nombre_propio(raw)
    if nombre:
        raw = re.sub(rf"^{re.escape(nombre)}\s*,\s*", "", raw).strip()

    raw = re.split(
        r"\b(?:searching for|looking for|trying to|lost in|travelling through|wandering)\b",
        raw, maxsplit=1, flags=re.IGNORECASE,
    )[0].strip(" ,.;")

    if not re.match(r"^(?:a|an|single|one)\b", raw, flags=re.IGNORECASE):
        raw = f"single {raw}"
    else:
        raw = re.sub(r"^(?:a|an)\b", "single", raw, count=1, flags=re.IGNORECASE)

    return " ".join(raw.split()[:14]), nombre


def obtener_accion(escena: dict, nombre_personaje: str) -> tuple:
    """(accion, es_generica). Sustituye el nombre propio por 'the character'."""
    accion = limpiar_prompt(escena.get("accion_en", ""))
    if not accion:
        return "exploring the environment carefully", True

    generica = accion.lower().startswith("main character performing")
    accion = re.split(r"(?<=[.!?])\s+", accion)[0]
    if nombre_personaje:
        accion = re.sub(rf"\b{re.escape(nombre_personaje)}\b", "the character", accion, flags=re.IGNORECASE)
    return " ".join(accion.split()[:18]), generica


def obtener_lugar(escena: dict, datos_historia: dict) -> tuple:
    """(lugar, es_global). Prefiere el lugar propio de la escena."""
    lugar = limpiar_prompt(escena.get("lugar_en", ""))
    if lugar:
        return " ".join(lugar.split()[:10]), False
    global_ = limpiar_prompt(datos_historia.get("escenario_en", "atmospheric world"))
    return " ".join(global_.split()[:10]), True


def ajustar_prompt_a_clip(tokenizer, estilo, plano, personaje, accion, lugar) -> str:
    """
    Orden: estilo (corto) -> plano -> personaje fijo -> acción de la escena -> lugar de la escena.
    Si no cabe en 77 tokens se recorta primero el lugar, luego el estilo, luego la acción
    y por último el personaje.
    """
    limite = min(MAX_TOKENS_PROMPT, tokenizer.model_max_length)
    partes = [limpiar_prompt(p) for p in (estilo, plano, personaje, accion, lugar)]

    def construir(lista):
        return ", ".join(p for p in lista if p)

    prompt = construir(partes)
    while contar_tokens_clip(tokenizer, prompt) > limite:
        if len(partes[4].split()) > 4:
            partes[4] = " ".join(partes[4].split()[:-1])
        elif len(partes[0].split()) > 5:
            partes[0] = " ".join(partes[0].split()[:-1])
        elif len(partes[3].split()) > 6:
            partes[3] = " ".join(partes[3].split()[:-1])
        elif len(partes[2].split()) > 6:
            partes[2] = " ".join(partes[2].split()[:-1])
        else:
            break
        prompt = construir(partes)

    if contar_tokens_clip(tokenizer, prompt) > limite:
        tokens = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        prompt = tokenizer.decode(tokens[:max(1, limite - 2)], skip_special_tokens=True).strip(" ,.;")
    return prompt


# ============================================================
# VALIDACIÓN Y CACHÉ ATÓMICA
# ============================================================

def validar_historia(datos_historia: dict):
    if not isinstance(datos_historia, dict):
        raise TypeError("datos_historia debe ser un diccionario.")
    escenas = datos_historia.get("escenas")
    if not isinstance(escenas, list) or len(escenas) != NUM_ESCENAS:
        raise ValueError(f"La historia debe contener exactamente {NUM_ESCENAS} escenas.")
    for idx, escena in enumerate(escenas, start=1):
        if not isinstance(escena, dict):
            raise ValueError(f"La escena {idx} no es válida.")


def _clave_cache(prompt: str, semilla: int, referencia: str = "") -> str:
    datos = {
        "prompt": prompt,
        "semilla": semilla,
        "resolucion": RESOLUCION,
        "pasos": PASOS_INFERENCIA,
        "guidance": GUIDANCE_SCALE,
        "modo": MODO_CONSISTENCIA,
        "strength": STRENGTH_IMG2IMG if referencia else None,
        "referencia": referencia,
        "modelo": os.path.abspath(RUTA_MODELO),
        "version_cache": 4,
    }
    return hashlib.sha256(json.dumps(datos, sort_keys=True).encode("utf-8")).hexdigest()


def _imagen_cache_valida(ruta: str, clave: str) -> bool:
    if not USAR_CACHE or not os.path.isfile(ruta):
        return False
    try:
        with open(ruta + ".json", "r", encoding="utf-8") as archivo:
            if json.load(archivo).get("clave") != clave:
                return False
        with Image.open(ruta) as imagen:
            imagen.verify()
        return True
    except (OSError, ValueError, KeyError):
        return False


def _guardar_imagen_atomica(imagen, ruta: str, meta: dict):
    carpeta = os.path.dirname(os.path.abspath(ruta))
    fd, temporal = tempfile.mkstemp(prefix=".storybook_", suffix=".png", dir=carpeta)
    os.close(fd)
    try:
        imagen.save(temporal, format="PNG")
        os.replace(temporal, ruta)
    finally:
        if os.path.exists(temporal):
            os.unlink(temporal)

    fd_meta, temporal_meta = tempfile.mkstemp(prefix=".storybook_", suffix=".json", dir=carpeta)
    try:
        with os.fdopen(fd_meta, "w", encoding="utf-8") as archivo:
            json.dump(meta, archivo, ensure_ascii=False)
        os.replace(temporal_meta, ruta + ".json")
    finally:
        if os.path.exists(temporal_meta):
            os.unlink(temporal_meta)


# ============================================================
# GENERADOR PRINCIPAL
# ============================================================

def generar_imagenes_escenas(
    datos_historia: dict,
    genero: str,
    carpeta_salida: str,
    callback_progreso=None,
) -> tuple:

    validar_historia(datos_historia)
    if MODO_CONSISTENCIA not in MODOS_VALIDOS:
        raise ValueError(f"STORYBOOK_IMG_MODO debe ser uno de {MODOS_VALIDOS}.")
    inicio = time.perf_counter()

    if not os.path.isdir(RUTA_MODELO):
        raise FileNotFoundError(f"No se encontró SD-Turbo en: {RUTA_MODELO}")

    os.makedirs(carpeta_salida, exist_ok=True)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dispositivo == "cuda" else torch.float32

    genero_norm, estilo = resolver_estilo(genero)
    personaje, nombre_personaje = obtener_personaje(datos_historia)
    titulo = datos_historia.get("titulo", "Storybook")
    semilla_base = crear_semilla(f"{titulo}|{personaje}|{genero_norm}")

    rutas, registro = [], []
    pipe = pipe_i2i = None
    referencia_ruta, referencia_clave = None, ""

    print("=" * 55)
    print("[IMÁGENES] SD-Turbo | modo de consistencia:", MODO_CONSISTENCIA)
    print("[IMÁGENES] Dispositivo:", dispositivo)
    print(f"[IMÁGENES] Género resuelto: {genero_norm} -> {estilo}")
    print("[IMÁGENES] Sujeto fijo:", personaje)
    print("[IMÁGENES] Semilla base:", semilla_base)
    print("=" * 55)

    try:
        argumentos = {"dtype": dtype, "local_files_only": True}
        if dispositivo == "cuda":
            argumentos["variant"] = "fp16"
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(RUTA_MODELO, **argumentos)
        except (OSError, ValueError):
            argumentos.pop("variant", None)
            pipe = AutoPipelineForText2Image.from_pretrained(RUTA_MODELO, **argumentos)

        pipe = pipe.to(dispositivo)
        pipe.set_progress_bar_config(disable=True)
        tokenizer = pipe.tokenizer
        limite_real = tokenizer.model_max_length
        tokens_fijos = contar_tokens_clip(tokenizer, f"{estilo}, {personaje}")
        print(f"[IMÁGENES] Tokens fijos (estilo+personaje): {tokens_fijos}/{limite_real}")

        for indice, escena in enumerate(datos_historia["escenas"], start=1):
            inicio_escena = time.perf_counter()
            accion, accion_generica = obtener_accion(escena, nombre_personaje)
            lugar, lugar_global = obtener_lugar(escena, datos_historia)
            plano = PLANOS[(indice - 1) % len(PLANOS)]

            if accion_generica:
                print(f"[IMÁGENES] AVISO escena {indice}: acción genérica; la imagen no reflejará la escena.")
            if lugar_global:
                print(f"[IMÁGENES] AVISO escena {indice}: sin lugar propio; se usa el escenario global.")

            prompt = ajustar_prompt_a_clip(tokenizer, estilo, plano, personaje, accion, lugar)
            cantidad_tokens = contar_tokens_clip(tokenizer, prompt)
            print(f"\n[IMÁGENES] Escena {indice}/5 | Tokens: {cantidad_tokens}/{limite_real}")
            print(f"[IMÁGENES] Prompt: {prompt}")

            semilla = semilla_para_escena(semilla_base, indice)
            usa_i2i = MODO_CONSISTENCIA == "img2img" and indice > 1 and referencia_ruta is not None
            ruta = os.path.join(carpeta_salida, f"escena_{indice}.png")
            clave = _clave_cache(prompt, semilla, referencia_clave if usa_i2i else "")

            if _imagen_cache_valida(ruta, clave):
                print("[IMÁGENES] Reutilizada desde caché:", ruta)
            else:
                generador = torch.Generator(device=dispositivo).manual_seed(semilla)
                with torch.inference_mode():
                    if usa_i2i:
                        if pipe_i2i is None:
                            pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe)
                            pipe_i2i.set_progress_bar_config(disable=True)
                        with Image.open(referencia_ruta) as ref:
                            ref = ref.convert("RGB")
                        resultado = pipe_i2i(
                            prompt=prompt, image=ref, strength=STRENGTH_IMG2IMG,
                            num_inference_steps=PASOS_INFERENCIA, guidance_scale=GUIDANCE_SCALE,
                            generator=generador,
                        )
                    else:
                        resultado = pipe(
                            prompt=prompt, num_inference_steps=PASOS_INFERENCIA,
                            guidance_scale=GUIDANCE_SCALE, width=RESOLUCION, height=RESOLUCION,
                            generator=generador,
                        )
                imagen = resultado.images[0]
                _guardar_imagen_atomica(imagen, ruta, {
                    "clave": clave, "prompt": prompt, "semilla": semilla, "modo": MODO_CONSISTENCIA,
                })
                del resultado, imagen, generador
                print(f"[IMÁGENES] Guardada: {ruta} ({round(time.perf_counter() - inicio_escena, 2)}s)")

            rutas.append(ruta)
            if indice == 1:
                referencia_ruta, referencia_clave = ruta, clave
            registro.append({
                "escena": indice, "prompt": prompt, "semilla": semilla, "plano": plano,
                "accion_generica": accion_generica, "lugar_global": lugar_global, "img2img": usa_i2i,
            })
            if callback_progreso is not None:
                callback_progreso(indice, NUM_ESCENAS)

        tiempo_total = round(time.perf_counter() - inicio, 2)
        with open(os.path.join(carpeta_salida, "prompts_usados.json"), "w", encoding="utf-8") as archivo:
            json.dump({"modo": MODO_CONSISTENCIA, "semilla_base": semilla_base, "personaje": personaje,
                       "estilo": estilo, "tiempo_s": tiempo_total, "escenas": registro},
                      archivo, ensure_ascii=False, indent=2)
        print("\n" + "=" * 55)
        print(f"[IMÁGENES] Completado: {len(rutas)} imágenes en {tiempo_total}s")
        print("=" * 55)
        return rutas, tiempo_total

    finally:
        if pipe_i2i is not None:
            del pipe_i2i
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[IMÁGENES] Memoria liberada.")