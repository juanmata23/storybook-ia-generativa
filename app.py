import os
import time
import gradio as gr
from modulos.generador_texto import generar_historia
from modulos.generador_imagen import generar_imagenes_escenas
from modulos.generador_voz import generar_voces_escenas
from modulos.generador_sonido import generar_sonidos_escenas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(BASE_DIR, "salida_cuento")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

def crear_storybook(idea, genero, progress=gr.Progress()):
    # Validación obligatoria de entradas no vacías
    if not idea or not idea.strip():
        raise gr.Error("Por favor, escribe una idea inicial para la historia.")
    if not genero:
        raise gr.Error("Por favor, selecciona un género literario.")

    inicio_total = time.time()

    progress(0.1, desc="1/4 Generando guion de 5 escenas (Qwen3)...")
    datos = generar_historia(idea.strip(), genero)

    progress(0.35, desc="2/4 Ilustrando las 5 escenas (SD-Turbo)...")
    imagenes = generar_imagenes_escenas(datos, genero, CARPETA_SALIDA)

    progress(0.60, desc="3/4 Sintetizando narración en español (Piper TTS)...")
    voces = generar_voces_escenas(datos, CARPETA_SALIDA)

    progress(0.80, desc="4/4 Componiendo ambientación sonora (MusicGen)...")
    sonidos = generar_sonidos_escenas(datos, genero, CARPETA_SALIDA)

    # Empaquetar las 5 escenas en el estado de la sesión
    escenas_completas = []
    for i in range(5):
        escenas_completas.append({
            "numero": i + 1,
            "titulo": datos["titulo"],
            "texto": datos["escenas"][i]["texto"],
            "imagen": imagenes[i],
            "voz": voces[i],
            "sonido": sonidos[i]
        })

    tiempo_total = time.time() - inicio_total
    progress(1.0, desc="¡Cuento generado!")

    primera = escenas_completas[0]
    encabezado = f"## 📖 {primera['titulo']}\n**Escena 1 de 5** *(Generado localmente en {tiempo_total:.1f}s)*"

    return (
        escenas_completas,
        0,  # índice actual (Escena 1)
        encabezado,
        primera["imagen"],
        primera["texto"],
        primera["voz"],
        primera["sonido"]
    )

def navegar_escena(escenas, indice_actual, direccion):
    if not escenas:
        return 0, "", None, "", None, None

    nuevo_indice = max(0, min(4, indice_actual + direccion))
    escena = escenas[nuevo_indice]
    encabezado = f"## 📖 {escena['titulo']}\n**Escena {escena['numero']} de 5**"

    return (
        nuevo_indice,
        encabezado,
        escena["imagen"],
        escena["texto"],
        escena["voz"],
        escena["sonido"]
    )

with gr.Blocks(title="Storybook IA Local - USTA Tunja") as demo:
    gr.Markdown("# 📚 Storybook: Generador Local de Cuentos Multimodales")
    gr.Markdown("Ingresa una idea y selecciona un género para generar 5 escenas con ilustración, narración y ambientación sonora 100% offline.")

    estado_escenas = gr.State([])
    estado_indice = gr.State(0)

    # Únicamente 2 entradas según especificación del proyecto
    with gr.Row():
        entrada_idea = gr.Textbox(
            label="Idea inicial de la historia",
            placeholder="Ej: Un gato astronauta descubre un planeta hecho de cristales musicales...",
            lines=2,
            scale=3
        )
        entrada_genero = gr.Dropdown(
            label="Tipo de cuento / Género",
            choices=["Infantil", "Fantasía", "Ciencia Ficción", "Terror", "Misterio", "Aventura"],
            value="Ciencia Ficción",
            scale=1
        )

    btn_generar = gr.Button("✨ Generar Cuento de 5 Escenas", variant="primary")

    gr.Markdown("---")
    salida_encabezado = gr.Markdown("### Aquí aparecerá tu cuento generado")

    with gr.Row():
        with gr.Column(scale=1):
            salida_imagen = gr.Image(label="Ilustración de la escena", type="filepath", height=400)
        with gr.Column(scale=1):
            salida_texto = gr.Textbox(label="Narración de la escena", lines=6, interactive=False)
            salida_voz = gr.Audio(label="🗣️ Voz Narrativa (Piper TTS)", type="filepath")
            salida_sonido = gr.Audio(label="🎵 Ambientación Sonora (MusicGen)", type="filepath")

    with gr.Row():
        btn_anterior = gr.Button("⬅️ Escena Anterior")
        btn_siguiente = gr.Button("Escena Siguiente ➡️")

    btn_generar.click(
        fn=crear_storybook,
        inputs=[entrada_idea, entrada_genero],
        outputs=[estado_escenas, estado_indice, salida_encabezado, salida_imagen, salida_texto, salida_voz, salida_sonido]
    )

    btn_anterior.click(
        fn=lambda esc, idx: navegar_escena(esc, idx, -1),
        inputs=[estado_escenas, estado_indice],
        outputs=[estado_indice, salida_encabezado, salida_imagen, salida_texto, salida_voz, salida_sonido]
    )

    btn_siguiente.click(
        fn=lambda esc, idx: navegar_escena(esc, idx, 1),
        inputs=[estado_escenas, estado_indice],
        outputs=[estado_indice, salida_encabezado, salida_imagen, salida_texto, salida_voz, salida_sonido]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)