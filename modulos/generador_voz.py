"""Puente de Storybook a Qwen3-TTS en un Python independiente (offline) con respaldo local."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import wave

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_TTS = Path(r"C:\Users\macaj\qwen3-tts-webui\.venv_prueba\Scripts\python.exe")
EJECUTOR = BASE_DIR / "modulos" / "ejecutor_qwen_tts.py"
RUTA_PIPER_RESPALDO = (
    BASE_DIR / "modelos" / "piper_voz" / "es" / "es_MX" / "claude" / "high" / "es_MX-claude-high.onnx"
)


def _generar_con_piper_respaldo(
    datos_historia: dict,
    salida: Path,
    inicio: float,
    callback_progreso=None
) -> tuple[list[str], float]:
    """Genera las narraciones localmente usando Piper TTS si Qwen3-TTS falla."""
    from piper.voice import PiperVoice

    if not RUTA_PIPER_RESPALDO.is_file():
        raise FileNotFoundError(
            f"Falló Qwen3-TTS y tampoco se encontró el modelo de respaldo Piper en: {RUTA_PIPER_RESPALDO}"
        )

    voz = PiperVoice.load(str(RUTA_PIPER_RESPALDO))
    escenas = datos_historia["escenas"]
    rutas = []

    for idx, escena in enumerate(escenas, start=1):
        ruta_wav = salida / f"escena_{idx}_voz.wav"
        with wave.open(str(ruta_wav), "wb") as archivo_wav:
            voz.synthesize_wav(escena["texto"].strip(), archivo_wav)
        rutas.append(str(ruta_wav))
        if callback_progreso:
            callback_progreso(idx, len(escenas))

    return rutas, round(time.perf_counter() - inicio, 2)


def generar_voces_escenas(
    datos_historia: dict,
    carpeta_salida: str,
    callback_progreso=None
) -> tuple[list[str], float]:
    inicio = time.perf_counter()
    escenas = datos_historia.get("escenas")
    if not isinstance(escenas, list) or not escenas:
        raise ValueError("La historia no contiene escenas.")

    salida = Path(carpeta_salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    if not PYTHON_TTS.is_file() or not EJECUTOR.is_file():
        print("[VOZ] Ejecutor Qwen3-TTS no disponible; usando Piper TTS local.", flush=True)
        return _generar_con_piper_respaldo(datos_historia, salida, inicio, callback_progreso)

    temporal = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False, dir=salida
        ) as archivo:
            json.dump(datos_historia, archivo, ensure_ascii=False)
            temporal = archivo.name

        # Aislar completamente el subproceso para que no herede PYTHONPATH de Streamlit
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1"
        })

        comando = [str(PYTHON_TTS), str(EJECUTOR), "--entrada", temporal, "--salida", str(salida)]
        resultado = None
        errores = []

        with subprocess.Popen(
            comando,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        ) as proceso:
            assert proceso.stdout is not None
            for linea in proceso.stdout:
                linea = linea.strip()
                if linea.startswith("PROGRESO:"):
                    try:
                        actual, total = map(int, linea.partition(":")[2].split("/"))
                        if callback_progreso:
                            callback_progreso(actual, total)
                    except ValueError:
                        errores.append(linea)
                elif linea.startswith("RESULTADO:"):
                    resultado = json.loads(linea.partition(":")[2])
                elif linea:
                    errores.append(linea)
                    print("[Qwen3-TTS]", linea, flush=True)

            codigo = proceso.wait()

        if codigo != 0 or resultado is None:
            print("[VOZ] Error en subproceso Qwen3-TTS; activando respaldo Piper TTS:", flush=True)
            print("\n".join(errores[-10:]), flush=True)
            return _generar_con_piper_respaldo(datos_historia, salida, inicio, callback_progreso)

        rutas = resultado["rutas"]
        if len(rutas) != len(escenas) or not all(Path(r).is_file() for r in rutas):
            print("[VOZ] Archivos incompletos en Qwen3-TTS; activando respaldo Piper TTS.", flush=True)
            return _generar_con_piper_respaldo(datos_historia, salida, inicio, callback_progreso)

        return rutas, round(time.perf_counter() - inicio, 2)

    finally:
        if temporal is not None:
            Path(temporal).unlink(missing_ok=True)