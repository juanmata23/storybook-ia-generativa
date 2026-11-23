\#  Storybook — Generador Multimodal Local de Cuentos Ilustrados



\*\*Universidad Santo Tomás – Seccional Tunja\*\*  

Facultad de Ingeniería de Sistemas • Electiva Ingeniería Aplicada: Inteligencia Artificial Generativa (2026)



\*\*Storybook\*\* es una aplicación de inteligencia artificial generativa \*\*100 % local y offline\*\* que, a partir de únicamente dos entradas del usuario (\*\*idea inicial\*\* y \*\*género literario\*\*), produce de forma automática y coherente un título y exactamente \*\*cinco escenas\*\* ilustradas, narradas en español y acompañadas de ambientación sonora independiente, presentadas en un visor interactivo tipo libro digital con mezcla simultánea de audio.



\---



\##  Características Principales



\* \*\*Ejecución 100 % Local y Offline:\*\* Opera sin APIs de pago ni conexión a Internet tras descargar los pesos de los modelos (`HF\_HUB\_OFFLINE=1`, `TRANSFORMERS\_OFFLINE=1`).

\* \*\*Estructura Narrativa en Cinco Actos:\*\* Genera relatos completos en español neutro (`Inicio`, `Desarrollo`, `Conflicto`, `Clímax` y `Desenlace`) mediante planificación en 3 pasos JSON, memoria acumulativa de hechos por escena y validación automática de idioma, longitud y no repetición.

\* \*\*Consistencia Visual Multimodal:\*\* Ancla una descripción física fija del protagonista (`personaje\_en`), adapta el estilo artístico al género literario, alterna cinco planos cinematográficos cíclicos con escenarios propios por escena (`lugar\_en`) y protege el límite de `75` tokens efectivos de CLIP.

\* \*\*Voz Sintética con Tolerancia a Fallos:\*\* Sintetiza la narración completa en español mediante \*\*Qwen3-TTS\*\* en subproceso aislado, con conmutación automática al respaldo local ultrarrápido \*\*Piper TTS\*\* (`< 2 s`) ante cualquier fallo de entorno.

\* \*\*Ambientación Sonora Independiente:\*\* Genera pistas ambientales originales por escena con \*\*MusicGen Small\*\* y las reproduce simultáneamente junto con la voz mediante \*\*Web Audio API\*\* con controles de volumen independientes.

\* \*\*Gestión Atómica de Memoria VRAM:\*\* Libera explícitamente la memoria de video (`gc.collect()` + `torch.cuda.empty\_cache()`) entre etapas para ejecutar los cuatro modelos en una GPU de 8 GB de VRAM sin desbordamientos.



\---



\##  Estructura del Proyecto



```text

proyecto\_storybook/

├── app\_streamlit.py              # Interfaz principal de usuario (2 entradas + progreso + métricas)

├── pyproject.toml                # Definición de dependencias del proyecto

├── uv.lock                       # Bloqueo determinista de versiones

├── LICENSE                       # Licencia MIT del código fuente

├── README.md                     # Documentación técnica y guía de reproducción

├── .gitignore                    # Exclusión de pesos pesados, entornos virtuales y temporales

├── modulos/

│   ├── \_\_init\_\_.py

│   ├── generador\_texto.py        # Qwen3-4B (NF4 4-bit): ficha, plan 5 actos, hechos y prompts

│   ├── generador\_imagen.py       # SD-Turbo (FP16): anclaje sujeto, planos cíclicos y caché SHA-256

│   ├── generador\_voz.py          # Puente aislado a Qwen3-TTS con respaldo automático Piper TTS

│   ├── ejecutor\_qwen\_tts.py      # Subproceso independiente para síntesis Qwen3-TTS

│   ├── generador\_sonido.py       # MusicGen Small (FP16): paisajes sonoros ambientales por escena

│   └── visor\_libro.py            # Visor HTML5/CSS3/JS de doble página con Web Audio API

├── evidencias/                   # Capturas y registros JSON de las 3 pruebas documentadas

└── modelos/                      # (No incluido en Git) Carpeta local de pesos preentrenados

&#x20;   ├── qwen3\_4b/

&#x20;   ├── sd\_turbo/

&#x20;   ├── musicgen\_small/

&#x20;   └── piper\_voz/

