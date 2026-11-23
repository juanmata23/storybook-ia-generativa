"""Narración offline de las escenas de Storybook con Qwen3-TTS Base.

Requisitos: torch (CUDA recomendado), qwen-tts, huggingface_hub, numpy,
soundfile. El modelo y los archivos de referencia deben existir localmente.

Optimizaciones respecto a la versión anterior:
- El prompt de la voz de referencia se calcula UNA vez (create_voice_clone_prompt)
  en lugar de reprocesar narrador.wav en cada fragmento.
- Los fragmentos de todas las escenas se sintetizan por lotes (TAMANO_LOTE).
- Tope de max_new_tokens según la longitud del texto (evita generaciones sin fin).
- Métricas de tiempo (carga, prompt, generación, factor de tiempo real).

Variables de entorno opcionales:
  STORYBOOK_TTS_MODELO  p. ej. Qwen/Qwen3-TTS-12Hz-0.6B-Base (más rápido; debe estar descargado)
  STORYBOOK_TTS_XVECTOR 1 = solo embedding del hablante (más rápido, menos fiel a la voz)
  STORYBOOK_TTS_LOTE    fragmentos por lote (por defecto 16; bájalo si la VRAM dedicada se llena)
  STORYBOOK_TTS_MAXCHARS caracteres máximos por fragmento (por defecto 240)
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

# Deben establecerse ANTES de importar Hugging Face / Transformers.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCIA_DIR = BASE_DIR / "modelos" / "voz_referencia"
REFERENCIA_WAV = REFERENCIA_DIR / "narrador.wav"
REFERENCIA_TXT = REFERENCIA_DIR / "narrador.txt"
MODELO_ID = os.environ.get("STORYBOOK_TTS_MODELO", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
SOLO_XVECTOR = os.environ.get("STORYBOOK_TTS_XVECTOR", "0") == "1"
TAMANO_LOTE = max(1, int(os.environ.get("STORYBOOK_TTS_LOTE", "16")))
MAX_CARACTERES_FRAGMENTO = int(os.environ.get("STORYBOOK_TTS_MAXCHARS", "240"))
IDIOMA = "Spanish"
PAUSA_ENTRE_FRAGMENTOS_S = 0.28

# Se llena al terminar; main() lo imprime en RESULTADO para el informe técnico.
ULTIMAS_METRICAS: dict = {}


def _partir_texto(texto: str, max_caracteres: int = 450) -> list[str]:
    """Divide textos largos respetando preferentemente los límites de frase."""
    texto = re.sub(r"\s+", " ", texto).strip()
    if not texto:
        return []
    frases = re.split(r"(?<=[.!?…])\s+", texto)
    partes: list[str] = []
    actual = ""
    for frase in frases:
        if len(frase) > max_caracteres:
            if actual:
                partes.append(actual)
                actual = ""
            palabras = frase.split()
            segmento = ""
            for palabra in palabras:
                candidato = f"{segmento} {palabra}".strip()
                if len(candidato) > max_caracteres and segmento:
                    partes.append(segmento)
                    segmento = palabra
                else:
                    segmento = candidato
            if segmento:
                partes.append(segmento)
            continue
        candidato = f"{actual} {frase}".strip()
        if len(candidato) > max_caracteres and actual:
            partes.append(actual)
            actual = frase
        else:
            actual = candidato
    if actual:
        partes.append(actual)
    return partes


def _sintetizar_lote(modelo, prompt, fragmentos: list[str]):
    """Sintetiza varios fragmentos. Devuelve (lista de arrays float32, tasa).

    Intenta un solo lote; si falla (p. ej. falta de VRAM), reintenta uno por uno.
    """
    import numpy as np
    import torch

    # A ~12,5 tokens/s y ~15 caracteres/s se necesitan ~0,85 tokens por carácter;
    # 2,0 deja margen amplio y corta las generaciones que no logran parar.
    limite = int(max(len(f) for f in fragmentos) * 2.0) + 64

    def llamar(textos, con_limite=True):
        extra = {"max_new_tokens": limite} if con_limite else {}
        with torch.inference_mode():
            return modelo.generate_voice_clone(
                text=textos,
                language=[IDIOMA] * len(textos),
                voice_clone_prompt=prompt,
                **extra,
            )

    try:
        try:
            ondas, sr = llamar(fragmentos)
        except TypeError:  # esta versión de qwen-tts no acepta max_new_tokens
            ondas, sr = llamar(fragmentos, con_limite=False)
        if len(ondas) != len(fragmentos):
            raise RuntimeError("El lote devolvió una cantidad distinta de audios.")
    except (RuntimeError, ValueError, TypeError) as error:
        if len(fragmentos) == 1:
            raise
        print(f"[TTS] lote de {len(fragmentos)} falló ({error}); dividiendo a la mitad.", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        mitad = len(fragmentos) // 2
        izquierda, sr = _sintetizar_lote(modelo, prompt, fragmentos[:mitad])
        derecha, sr = _sintetizar_lote(modelo, prompt, fragmentos[mitad:])
        return izquierda + derecha, sr

    audios = []
    for onda in ondas:
        audio = np.asarray(onda, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("Qwen3-TTS devolvió un audio inválido.")
        audios.append(audio)
    return audios, sr


def _ensamblar(audios, tasa):
    import numpy as np

    piezas = []
    for i, audio in enumerate(audios):
        if i:
            piezas.append(np.zeros(int(tasa * PAUSA_ENTRE_FRAGMENTOS_S), dtype=np.float32))
        piezas.append(audio)
    final = np.concatenate(piezas)
    pico = float(np.max(np.abs(final)))
    if pico > 0.98:
        final = final * (0.98 / pico)
    return final


def _escribir_wav(wav: Path, meta: Path, clave: str, audio, tasa) -> None:
    import soundfile as sf

    temporal = wav.with_name(wav.stem + ".tmp.wav")
    try:
        sf.write(str(temporal), audio, tasa, subtype="PCM_16")
        os.replace(temporal, wav)
        meta.write_text(json.dumps({"clave": clave}, indent=2), encoding="utf-8")
    finally:
        temporal.unlink(missing_ok=True)


def generar_voces_escenas(
    datos_historia: dict,
    carpeta_salida: str,
    callback_progreso=None,
) -> tuple[list[str], float]:
    """Genera escena_N_voz.wav y devuelve (rutas, segundos).

    Los WAV existentes se reutilizan únicamente si el texto, la referencia y
    la configuración son idénticos a los de la generación anterior.
    """
    inicio = time.perf_counter()
    escenas = datos_historia.get("escenas")
    if not isinstance(escenas, list) or not escenas:
        raise ValueError("datos_historia debe contener una lista no vacía 'escenas'.")
    if not REFERENCIA_WAV.is_file() or not REFERENCIA_TXT.is_file():
        raise FileNotFoundError(
            f"Faltan narrador.wav o narrador.txt en {REFERENCIA_DIR}"
        )

    texto_referencia = REFERENCIA_TXT.read_text(encoding="utf-8-sig").strip()
    if not texto_referencia:
        raise ValueError("narrador.txt está vacío.")

    destino = Path(carpeta_salida)
    destino.mkdir(parents=True, exist_ok=True)
    textos = []
    for indice, escena in enumerate(escenas, 1):
        if not isinstance(escena, dict) or not str(escena.get("texto", "")).strip():
            raise ValueError(f"La escena {indice} no contiene un texto válido.")
        textos.append(str(escena["texto"]).strip())

    referencia_hash = hashlib.sha256(REFERENCIA_WAV.read_bytes()).hexdigest()
    rutas: list[str] = []
    pendientes = []
    for indice, texto in enumerate(textos, 1):
        wav = destino / f"escena_{indice}_voz.wav"
        meta = destino / f"escena_{indice}_voz.meta.json"
        clave = hashlib.sha256(json.dumps({
            "modelo": MODELO_ID,
            "idioma": IDIOMA,
            "texto": texto,
            "referencia_audio_sha256": referencia_hash,
            "referencia_texto": texto_referencia,
            "solo_xvector": SOLO_XVECTOR,
            "max_caracteres": MAX_CARACTERES_FRAGMENTO,
            "version_modulo": 3,
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        rutas.append(str(wav))
        try:
            cache_valida = wav.is_file() and wav.stat().st_size > 44 and json.loads(
                meta.read_text(encoding="utf-8")
            ).get("clave") == clave
        except (OSError, ValueError, TypeError):
            cache_valida = False
        if not cache_valida:
            pendientes.append((indice, texto, wav, meta, clave))

    if not pendientes:
        ULTIMAS_METRICAS.clear()
        ULTIMAS_METRICAS.update({"cache": True})
        if callback_progreso:
            for indice in range(1, len(escenas) + 1):
                callback_progreso(indice, len(escenas))
        return rutas, round(time.perf_counter() - inicio, 2)

    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No se detectó CUDA. Qwen3-TTS requiere mucha memoria en CPU; "
            "verifica PyTorch CUDA y el controlador NVIDIA."
        )

    # Sin red: si faltan pesos en caché, snapshot_download falla inmediatamente.
    ruta_modelo = snapshot_download(repo_id=MODELO_ID, local_files_only=True)
    modelo = None
    try:
        t0 = time.perf_counter()
        modelo = Qwen3TTSModel.from_pretrained(
            ruta_modelo,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        t_carga = time.perf_counter() - t0

        # Se calcula una sola vez y se reutiliza en todos los fragmentos.
        t1 = time.perf_counter()
        prompt = modelo.create_voice_clone_prompt(
            ref_audio=str(REFERENCIA_WAV),
            ref_text=texto_referencia,
            x_vector_only_mode=SOLO_XVECTOR,
        )
        t_prompt = time.perf_counter() - t1

        # Todos los fragmentos de todas las escenas pendientes, en orden.
        trabajo = []
        partes: dict[int, list] = {}
        destinos = {}
        for indice, texto, wav, meta, clave in pendientes:
            fragmentos = _partir_texto(texto, MAX_CARACTERES_FRAGMENTO)
            if not fragmentos:
                raise ValueError(f"La escena {indice} está vacía.")
            partes[indice] = [None] * len(fragmentos)
            destinos[indice] = (wav, meta, clave)
            trabajo += [(indice, k, f) for k, f in enumerate(fragmentos)]

        hechas = len(escenas) - len(pendientes)
        segundos_audio = 0.0
        tasa = None
        t2 = time.perf_counter()
        for i in range(0, len(trabajo), TAMANO_LOTE):
            lote = trabajo[i:i + TAMANO_LOTE]
            t_lote = time.perf_counter()
            audios, sr = _sintetizar_lote(modelo, prompt, [f for _, _, f in lote])
            if tasa is not None and sr != tasa:
                raise RuntimeError("Qwen3-TTS devolvió tasas de muestreo diferentes.")
            tasa = sr
            print(f"[TTS] lote {i // TAMANO_LOTE + 1}: {len(lote)} fragmentos en "
                  f"{time.perf_counter() - t_lote:.1f}s", flush=True)
            for (indice, k, _), audio in zip(lote, audios):
                partes[indice][k] = audio
            # Escribe cada escena en cuanto tiene todos sus fragmentos.
            for indice in list(partes):
                if all(p is not None for p in partes[indice]):
                    audio_final = _ensamblar(partes.pop(indice), tasa)
                    wav, meta, clave = destinos[indice]
                    _escribir_wav(wav, meta, clave, audio_final, tasa)
                    segundos_audio += len(audio_final) / tasa
                    hechas += 1
                    if callback_progreso:
                        callback_progreso(hechas, len(escenas))
        t_gen = time.perf_counter() - t2

        ULTIMAS_METRICAS.clear()
        ULTIMAS_METRICAS.update({
            "modelo": MODELO_ID,
            "solo_xvector": SOLO_XVECTOR,
            "tamano_lote": TAMANO_LOTE,
            "max_caracteres": MAX_CARACTERES_FRAGMENTO,
            "carga_modelo_s": round(t_carga, 2),
            "prompt_voz_s": round(t_prompt, 2),
            "generacion_s": round(t_gen, 2),
            "audio_generado_s": round(segundos_audio, 2),
            "factor_tiempo_real": round(t_gen / segundos_audio, 3) if segundos_audio else None,
        })
        print(f"[TTS] {ULTIMAS_METRICAS}", flush=True)
    finally:
        del modelo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rutas, round(time.perf_counter() - inicio, 2)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generador de narraciones Qwen3-TTS offline")
    parser.add_argument("--entrada", required=True, help="JSON con datos_historia")
    parser.add_argument("--salida", required=True, help="Carpeta de WAV")
    args = parser.parse_args()
    with open(args.entrada, encoding="utf-8") as archivo:
        datos = json.load(archivo)
    rutas, duracion = generar_voces_escenas(
        datos, args.salida,
        callback_progreso=lambda actual, total: print(f"PROGRESO:{actual}/{total}", flush=True),
    )
    print("RESULTADO:" + json.dumps(
        {"rutas": rutas, "tiempo": duracion, "metricas": ULTIMAS_METRICAS}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()