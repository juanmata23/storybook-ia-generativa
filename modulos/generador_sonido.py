"""
STORYBOOK - GENERADOR DE AMBIENTE SONORO CON MUSICGEN (local)

MusicGen es un modelo de MÚSICA: entiende descripciones musicales (estilo, instrumentos, ánimo,
tempo) mucho mejor que descripciones literales de efectos ("wind howling, door creaking").
Por eso este módulo traduce la escena a una descripción musical:

    estilo del género + texturas derivadas del sonido de la escena + ánimo según el acto + "instrumental"

Mejoras respecto a la versión anterior:
- Prompt musical estructurado (antes: plantilla genérica que ahogaba el sonido propio de la escena).
- Texturas musicales por palabras clave (viento -> pads aireados, tormenta -> timbales graves, etc.).
- Arco emocional: el ánimo/tempo cambia con el acto (inicio, desarrollo, conflicto, clímax, desenlace).
- Clips de ~12 s (antes 6 s) que pueden extenderse en bucle con fundido cruzado hasta la duración
  de la narración (parámetro opcional `duraciones_s`).
- Generación en LOTE (las 5 escenas en una sola llamada), fp16 en GPU y modo offline.
- Fundidos, normalización a nivel bajo (para que la voz domine) y WAV de 16 bits.
- Caché con metadatos, carpeta creada si no existe y registro de prompts (sonidos_usados.json).

Variables de entorno opcionales:
  STORYBOOK_SND_SEGUNDOS  duración del clip generado (por defecto 12; máximo recomendado 30)
  STORYBOOK_SND_GUIDANCE  fidelidad al texto (por defecto 3.5; más alto = más literal, menos variado)
  STORYBOOK_SND_FP16      1 = fp16 en GPU (por defecto); 0 = fp32 si el audio sale vacío o con ruido
"""

import gc
import hashlib
import json
import math
import os
import re
import tempfile
import time
import unicodedata
import wave

import numpy as np
import scipy.io.wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration, set_seed

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "musicgen_small")

DURACION_CLIP_S = min(30.0, float(os.environ.get("STORYBOOK_SND_SEGUNDOS", "12")))
TOKENS_POR_SEGUNDO = 50  # MusicGen: 50 tokens de audio por segundo
GUIDANCE_SCALE = float(os.environ.get("STORYBOOK_SND_GUIDANCE", "3.5"))
USAR_FP16 = os.environ.get("STORYBOOK_SND_FP16", "1") == "1"
USAR_CACHE = True

PICO_OBJETIVO = 0.5          # el ambiente queda ~6 dB por debajo del máximo: la voz debe dominar
FADE_ENTRADA_S = 0.5
FADE_SALIDA_S = 1.2
CRUCE_BUCLE_S = 1.5
MARGEN_SOBRE_VOZ_S = 1.0     # el ambiente dura un poco más que la narración

# ============================================================
# ESTILO MUSICAL POR GÉNERO / ÁNIMO POR ACTO / TEXTURAS POR PALABRA CLAVE
# ============================================================

ESTILOS_GENERO = {
    "infantil": "playful whimsical children's music, music box, glockenspiel, ukulele, light pizzicato strings",
    "fantasia": "orchestral fantasy music, harp, flute, soft strings, magical chimes",
    "ciencia ficcion": "ambient electronic music, analog synth pads, subtle arpeggios, spacey reverb",
    "terror": "dark ambient horror music, low drones, eerie strings, sparse piano, dissonant tones",
    "misterio": "noir mystery ambient music, muted piano, soft double bass, subtle strings, suspenseful",
    "aventura": "adventurous orchestral music, warm strings, light percussion, soft brass swells",
}

# Un ánimo por acto (Inicio, Desarrollo, Conflicto, Clímax, Desenlace): da arco emocional a la banda sonora.
HUMORES_ACTO = [
    "calm and curious mood, slow tempo, gentle",
    "flowing and hopeful mood, moderate tempo, building warmth",
    "tense and uneasy mood, rising intensity, hesitant rhythm",
    "intense and dramatic mood, powerful, driving rhythm",
    "resolved and warm mood, slow tempo, peaceful ending",
]

# (patrón sobre la descripción en inglés, descriptor musical equivalente). Se usan los 2 primeros que coincidan.
TEXTURAS = [
    (r"thunder|storm|lightning", "deep timpani rumbles and dark rolling drones"),
    (r"rain|drizzle|drip", "soft shimmering textures like falling rain and gentle piano notes"),
    (r"wind|breeze|gust|howl", "airy pads with breathy flute and whistling wind-like textures"),
    (r"wave|ocean|sea\b|water|river|stream|lake", "flowing harp arpeggios and watery shimmering pads"),
    (r"fire|flame|campfire|crackl|torch|lantern", "warm crackling textures with low cello"),
    (r"footstep|walk|march|steps|run", "soft rhythmic pizzicato strings with a light steady pulse"),
    (r"forest|jungle|bird|leaves|trees|garden|meadow", "light flute, chimes and plucked strings with a natural feeling"),
    (r"cave|dungeon|tunnel|echo|underground", "deep echoing drones and sparse resonant low tones"),
    (r"door|creak|wooden|floor|house|library|attic|mansion", "creaking low strings and sparse muted piano notes"),
    (r"space|star|cosmic|planet|galax|alien|rocket|engine|machine|robot|computer|electric|ship", "spacey analog synth pads and pulsing arpeggios"),
    (r"night|owl|moon|midnight|dark|shadow|ghost", "low drone with celesta and sparse eerie tones"),
    (r"heartbeat|pulse|drum", "slow deep pulsing bass drum and rising tension"),
    (r"crowd|market|village|town|festival|party|celebrat", "lively folk strings with light percussion"),
    (r"magic|spell|sparkle|glow|crystal|mirror|shimmer", "sparkling chimes, glockenspiel and soft harp glissandos"),
    (r"snow|ice|frozen|winter|cold", "cold glassy bells and sustained high strings"),
    (r"desert|sand|dune", "sparse plucked strings and warm dry pads"),
]

PALABRAS_VOCALES = re.compile(
    r"\b(voices?|vocals?|speech|speaking|talking|singing|choir|whisper(?:s|ing)?|"
    r"laugh(?:s|ter|ing)?|scream(?:s|ing)?|shout(?:s|ing)?|human)\b", re.I)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_clave(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z\s]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def resolver_genero(genero: str) -> str:
    clave = normalizar_clave(genero)
    if "ciencia" in clave or "ficc" in clave:
        return "ciencia ficcion"
    if "terror" in clave or "horror" in clave:
        return "terror"
    if "misterio" in clave:
        return "misterio"
    if "aventura" in clave:
        return "aventura"
    if "infantil" in clave:
        return "infantil"
    return "fantasia"


def limpiar_descripcion(desc) -> str:
    """Quita palabras de voz humana (MusicGen podría generarlas) y limita la longitud."""
    desc = PALABRAS_VOCALES.sub("", str(desc or ""))
    desc = re.sub(r"\s+", " ", desc).strip(" ,.;")
    return " ".join(desc.split()[:10])


def texturas_desde_descripcion(desc: str) -> list:
    encontradas = []
    for patron, descriptor in TEXTURAS:
        if re.search(r"\b(?:" + patron + ")", desc, flags=re.I) and descriptor not in encontradas:
            encontradas.append(descriptor)
        if len(encontradas) == 2:
            break
    return encontradas


def construir_prompt(genero_clave: str, numero: int, desc: str) -> str:
    texturas = texturas_desde_descripcion(desc)
    if not texturas and desc:
        texturas = [desc]  # sin coincidencias: se usa la descripción tal cual
    humor = HUMORES_ACTO[(numero - 1) % len(HUMORES_ACTO)]
    partes = [ESTILOS_GENERO[genero_clave], *texturas, humor, "instrumental, no vocals"]
    return ", ".join(partes)


def crear_semilla(texto: str) -> int:
    return int(hashlib.sha256(texto.encode("utf-8")).hexdigest()[:8], 16) % (2**32)


def duracion_wav(ruta: str):
    """Duración en segundos de un WAV PCM (voz de Qwen3-TTS o Piper); None si no se puede leer."""
    try:
        with wave.open(ruta, "rb") as archivo:
            return archivo.getnframes() / float(archivo.getframerate())
    except (OSError, wave.Error, EOFError):
        return None


def duraciones_desde_wavs(rutas_voz: list) -> list:
    return [duracion_wav(r) for r in rutas_voz]


def extender_en_bucle(audio: np.ndarray, sr: int, objetivo_s: float, cruce_s: float = CRUCE_BUCLE_S) -> np.ndarray:
    """Repite el clip con fundido cruzado de potencia constante hasta alcanzar objetivo_s."""
    objetivo = int(objetivo_s * sr)
    if len(audio) >= objetivo:
        return audio[:objetivo]
    cruce = max(1, min(int(cruce_s * sr), len(audio) // 3))
    entra = np.sin(np.linspace(0, np.pi / 2, cruce)).astype(np.float32)
    sale = np.cos(np.linspace(0, np.pi / 2, cruce)).astype(np.float32)
    salida = audio.copy()
    while len(salida) < objetivo:
        union = salida[-cruce:] * sale + audio[:cruce] * entra
        salida = np.concatenate([salida[:-cruce], union, audio[cruce:]])
    return salida[:objetivo]


def acabado_audio(audio: np.ndarray, sr: int, objetivo_s=None) -> np.ndarray:
    """Bucle opcional + fundidos + normalización de pico. Devuelve int16."""
    audio = audio.astype(np.float32)
    audio = audio - float(audio.mean())
    if objetivo_s:
        audio = extender_en_bucle(audio, sr, objetivo_s)
    fi = min(int(FADE_ENTRADA_S * sr), len(audio) // 4)
    fo = min(int(FADE_SALIDA_S * sr), len(audio) // 4)
    if fi > 0:
        audio[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        audio[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    pico = float(np.max(np.abs(audio)))
    if pico > 0:
        audio = audio * (PICO_OBJETIVO / pico)
    return (audio * 32767).astype(np.int16)


# ============================================================
# CACHÉ ATÓMICA
# ============================================================

def _clave_cache(prompt: str, semilla: int, objetivo_s) -> str:
    datos = {
        "prompt": prompt, "semilla": semilla, "segundos_clip": DURACION_CLIP_S,
        "guidance": GUIDANCE_SCALE, "objetivo_s": objetivo_s, "fp16": USAR_FP16,
        "modelo": os.path.abspath(RUTA_MODELO), "version_cache": 2,
    }
    return hashlib.sha256(json.dumps(datos, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_valida(ruta: str, clave: str) -> bool:
    if not USAR_CACHE or not os.path.isfile(ruta) or os.path.getsize(ruta) <= 44:
        return False
    try:
        with open(ruta + ".json", "r", encoding="utf-8") as archivo:
            return json.load(archivo).get("clave") == clave
    except (OSError, ValueError):
        return False


def _guardar_wav_atomico(ruta: str, sr: int, audio_int16: np.ndarray, meta: dict):
    carpeta = os.path.dirname(os.path.abspath(ruta))
    fd, temporal = tempfile.mkstemp(prefix=".storybook_", suffix=".wav", dir=carpeta)
    os.close(fd)
    try:
        scipy.io.wavfile.write(temporal, rate=sr, data=audio_int16)
        os.replace(temporal, ruta)
    finally:
        if os.path.exists(temporal):
            os.unlink(temporal)
    with open(ruta + ".json", "w", encoding="utf-8") as archivo:
        json.dump(meta, archivo, ensure_ascii=False)


# ============================================================
# GENERADOR PRINCIPAL
# ============================================================

def _generar_audios(modelo, procesador, prompts: list, dispositivo: str):
    """Genera todos los prompts en un lote; si no cabe en memoria, de uno en uno."""
    tokens = int(DURACION_CLIP_S * TOKENS_POR_SEGUNDO)

    def llamar(lista):
        entradas = procesador(text=lista, padding=True, return_tensors="pt").to(dispositivo)
        with torch.inference_mode():
            return modelo.generate(**entradas, do_sample=True, guidance_scale=GUIDANCE_SCALE,
                                   max_new_tokens=tokens)

    try:
        salida = llamar(prompts)
        audios = [salida[i, 0].float().cpu().numpy() for i in range(len(prompts))]
    except RuntimeError as error:
        if len(prompts) == 1:
            raise
        print(f"[SONIDO] lote falló ({error}); generando uno por uno.", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        audios = []
        for prompt in prompts:
            audios.append(llamar([prompt])[0, 0].float().cpu().numpy())

    for audio in audios:
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("MusicGen devolvió audio inválido. Prueba STORYBOOK_SND_FP16=0.")
    return audios


def generar_sonidos_escenas(
    datos_historia: dict,
    genero: str,
    carpeta_salida: str,
    callback_progreso=None,
    duraciones_s=None,
) -> tuple:
    """
    Genera escena_N_ambiente.wav. Devuelve (rutas, segundos).

    duraciones_s: lista opcional con la duración (s) de la narración de cada escena. Si se entrega,
    el ambiente se extiende en bucle (con fundido cruzado) hasta cubrirla; usa
    duraciones_desde_wavs(rutas_voz) para obtenerla. Sin ella, cada ambiente dura ~DURACION_CLIP_S.
    """
    inicio = time.perf_counter()
    escenas = datos_historia.get("escenas")
    if not isinstance(escenas, list) or not escenas:
        raise ValueError("datos_historia debe contener una lista no vacía 'escenas'.")
    if not os.path.isdir(RUTA_MODELO):
        raise FileNotFoundError(f"No se encontró MusicGen en: {RUTA_MODELO}")
    os.makedirs(carpeta_salida, exist_ok=True)

    genero_clave = resolver_genero(genero)
    titulo = datos_historia.get("titulo", "Storybook")
    semilla_base = crear_semilla(f"{titulo}|{genero_clave}")
    total = len(escenas)

    print("=" * 55)
    print(f"[SONIDO] MusicGen | género: {genero_clave} | clip: {DURACION_CLIP_S:g}s | guidance: {GUIDANCE_SCALE}")
    print("=" * 55)

    trabajos, registro = [], []
    for idx, escena in enumerate(escenas, start=1):
        desc = limpiar_descripcion(escena.get("sonido_en", "") if isinstance(escena, dict) else "")
        prompt = construir_prompt(genero_clave, idx, desc)
        objetivo = None
        if duraciones_s and idx - 1 < len(duraciones_s) and duraciones_s[idx - 1]:
            objetivo = float(math.ceil(duraciones_s[idx - 1])) + MARGEN_SOBRE_VOZ_S
        ruta = os.path.join(carpeta_salida, f"escena_{idx}_ambiente.wav")
        clave = _clave_cache(prompt, semilla_base, objetivo)
        trabajos.append({"idx": idx, "prompt": prompt, "ruta": ruta, "clave": clave, "objetivo": objetivo,
                         "cache": _cache_valida(ruta, clave)})
        registro.append({"escena": idx, "sonido_en": desc, "prompt": prompt, "objetivo_s": objetivo})
        print(f"[SONIDO] Escena {idx}/{total} | {'caché' if trabajos[-1]['cache'] else 'nuevo'} | {prompt}")

    pendientes = [t for t in trabajos if not t["cache"]]
    hechas = total - len(pendientes)
    if callback_progreso and hechas:
        callback_progreso(hechas, total)

    if pendientes:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        modelo = procesador = None
        try:
            t0 = time.perf_counter()
            procesador = AutoProcessor.from_pretrained(RUTA_MODELO, local_files_only=True)
            modelo = MusicgenForConditionalGeneration.from_pretrained(RUTA_MODELO, local_files_only=True)
            modelo = modelo.to(dispositivo)
            if dispositivo == "cuda" and USAR_FP16:
                modelo = modelo.half()
            modelo.eval()
            sr = modelo.config.audio_encoder.sampling_rate
            t_carga = time.perf_counter() - t0

            set_seed(semilla_base)
            t1 = time.perf_counter()
            audios = _generar_audios(modelo, procesador, [t["prompt"] for t in pendientes], dispositivo)
            t_gen = time.perf_counter() - t1
            print(f"[SONIDO] carga {t_carga:.1f}s | generación {t_gen:.1f}s para {len(pendientes)} clips")

            for trabajo, audio in zip(pendientes, audios):
                final = acabado_audio(audio, sr, trabajo["objetivo"])
                _guardar_wav_atomico(trabajo["ruta"], sr, final, {
                    "clave": trabajo["clave"], "prompt": trabajo["prompt"], "semilla": semilla_base,
                    "duracion_s": round(len(final) / sr, 2),
                })
                hechas += 1
                if callback_progreso:
                    callback_progreso(hechas, total)
        finally:
            del modelo, procesador
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[SONIDO] Memoria liberada.")

    tiempo_total = round(time.perf_counter() - inicio, 2)
    with open(os.path.join(carpeta_salida, "sonidos_usados.json"), "w", encoding="utf-8") as archivo:
        json.dump({"genero": genero_clave, "semilla": semilla_base, "guidance": GUIDANCE_SCALE,
                   "segundos_clip": DURACION_CLIP_S, "tiempo_s": tiempo_total, "escenas": registro},
                  archivo, ensure_ascii=False, indent=2)
    print(f"[SONIDO] Completado en {tiempo_total}s")
    return [t["ruta"] for t in trabajos], tiempo_total