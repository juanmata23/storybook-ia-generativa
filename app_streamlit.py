import os
import json
import time
import streamlit as st
import streamlit.components.v1 as components

from modulos.generador_texto import generar_historia
from modulos.generador_imagen import generar_imagenes_escenas
from modulos.generador_voz import generar_voces_escenas
from modulos.generador_sonido import generar_sonidos_escenas, duraciones_desde_wavs
from modulos.visor_libro import construir_html_libro


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(BASE_DIR, "salida_cuento")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

st.set_page_config(
    page_title="Storybook · Biblioteca Generativa Local",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Cormorant+Garamond:ital,wght@0,500;1,500&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

.stApp {
    background: radial-gradient(circle at 50% 0%, #131d33 0%, #090d16 65%, #060910 100%);
    color: #efe8d8;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.encabezado-biblioteca {
    text-align: center;
    padding: 12px 0 24px 0;
    border-bottom: 1px solid rgba(212, 175, 99, 0.25);
    margin-bottom: 28px;
}

.encabezado-biblioteca .sobretitulo {
    font-family: 'Cinzel', serif;
    font-size: 0.78rem;
    letter-spacing: 0.28em;
    color: #d4af63;
    text-transform: uppercase;
}

.encabezado-biblioteca h1 {
    font-family: 'Cinzel', serif;
    font-size: clamp(1.9rem, 3.4vw, 2.8rem);
    color: #fbf5e6;
    margin: 8px 0 6px 0;
    font-weight: 700;
}

.encabezado-biblioteca p {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.25rem;
    color: #bfb39b;
    margin: 0;
}

.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    background-color: #101829 !important;
    color: #f4eee1 !important;
    border: 1px solid rgba(212, 175, 99, 0.4) !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stTextArea label, .stSelectbox label {
    font-family: 'Cinzel', serif !important;
    color: #dfc07c !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.08em !important;
}

.stButton > button {
    background: linear-gradient(180deg, #dfc07c 0%, #b58938 100%) !important;
    color: #120f0a !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.72rem 2rem !important;
    width: 100%;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    transition: transform 0.2s ease, filter 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
}

.tarjeta-etapa {
    background: rgba(16, 24, 41, 0.85);
    border: 1px solid rgba(212, 175, 99, 0.32);
    border-radius: 10px;
    padding: 18px 22px;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

if "html_libro" not in st.session_state:
    st.session_state.html_libro = None
if "metricas" not in st.session_state:
    st.session_state.metricas = None
if "avisos" not in st.session_state:
    st.session_state.avisos = []

st.markdown("""
<div class="encabezado-biblioteca">
  <div class="sobretitulo">Taller Literario Multimodal · Ejecución Local</div>
  <h1>Storybook: Cuentos Ilustrados</h1>
  <p>Escribe una semilla narrativa y elige un género para dar vida a un libro ilustrado con voz y ambientación sonora.</p>
</div>
""", unsafe_allow_html=True)

col_idea, col_genero = st.columns([3, 1], gap="medium")

with col_idea:
    idea_usuario = st.text_area(
        "Idea inicial de la historia",
        placeholder="Ej: Un relojero solitario en una ciudad sumergida descubre que sus relojes pueden detener la lluvia...",
        height=96
    )

with col_genero:
    genero_usuario = st.selectbox(
        "Género literario",
        options=["Fantasía", "Ciencia Ficción", "Misterio", "Terror", "Aventura", "Infantil"]
    )

btn_crear = st.button("Crear mi historia")

if btn_crear:
    if not idea_usuario or not idea_usuario.strip():
        st.error("Por favor, escribe una idea inicial antes de comenzar la creación del libro.")
    elif not genero_usuario:
        st.error("Selecciona un género literario para continuar.")
    else:
        st.session_state.html_libro = None
        st.session_state.avisos = []
        inicio_global = time.perf_counter()

        contenedor_progreso = st.empty()
        barra = st.progress(0)

        try:
            contenedor_progreso.markdown(
                "<div class='tarjeta-etapa'><strong>Etapa 1 de 5 · Construyendo la historia</strong><br>"
                "Redactando el título, la biblia de personajes y las cinco escenas en español...</div>",
                unsafe_allow_html=True
            )
            datos_historia = generar_historia(
                idea_usuario.strip(),
                genero_usuario,
                callback_progreso=lambda actual, total: barra.progress(int((actual / total) * 30))
            )

            contenedor_progreso.markdown(
                "<div class='tarjeta-etapa'><strong>Etapa 2 de 5 · Creando las ilustraciones</strong><br>"
                "Pintando las cinco láminas con coherencia visual de personaje y escenario...</div>",
                unsafe_allow_html=True
            )
            rutas_imagenes, t_img = generar_imagenes_escenas(
                datos_historia,
                genero_usuario,
                CARPETA_SALIDA,
                callback_progreso=lambda actual, total: barra.progress(30 + int((actual / total) * 30))
            )

            contenedor_progreso.markdown(
                "<div class='tarjeta-etapa'><strong>Etapa 3 de 5 · Generando la narración</strong><br>"
                "Sintetizando la lectura completa en español para cada página...</div>",
                unsafe_allow_html=True
            )
            rutas_voces, t_voz = generar_voces_escenas(
                datos_historia,
                CARPETA_SALIDA,
                callback_progreso=lambda actual, total: barra.progress(60 + int((actual / total) * 15))
            )

            # Duración de cada narración: el ambiente sonoro se extiende para cubrirla.
            # Si algún WAV no se puede leer, devuelve None y ese ambiente dura solo el clip generado.
            duraciones_voz = duraciones_desde_wavs(rutas_voces)

            contenedor_progreso.markdown(
                "<div class='tarjeta-etapa'><strong>Etapa 4 de 5 · Preparando los sonidos</strong><br>"
                "Componiendo la ambientación sonora de cada escena...</div>",
                unsafe_allow_html=True
            )
            rutas_sonidos, t_son = generar_sonidos_escenas(
                datos_historia,
                genero_usuario,
                CARPETA_SALIDA,
                callback_progreso=lambda actual, total: barra.progress(75 + int((actual / total) * 20)),
                duraciones_s=duraciones_voz
            )

            contenedor_progreso.markdown(
                "<div class='tarjeta-etapa'><strong>Etapa 5 de 5 · Organizando el libro</strong><br>"
                "Encuadernando páginas, ilustraciones y pistas de audio...</div>",
                unsafe_allow_html=True
            )
            html_render = construir_html_libro(
                datos_historia, genero_usuario, rutas_imagenes, rutas_voces, rutas_sonidos
            )
            barra.progress(100)

            tiempo_total = round(time.perf_counter() - inicio_global, 2)
            palabras_totales = sum(len(e["texto"].split()) for e in datos_historia["escenas"])

            with open(os.path.join(CARPETA_SALIDA, "ultimo_cuento.json"), "w", encoding="utf-8") as f:
                json.dump(datos_historia, f, ensure_ascii=False, indent=2)

            metricas = {
                "total": tiempo_total,
                "texto": datos_historia.get("tiempo_texto_s", 0),
                "imagen": t_img,
                "voz": t_voz,
                "sonido": t_son,
                "palabras": palabras_totales
            }

            avisos = list(datos_historia.get("avisos", []))
            if datos_historia.get("uso_respaldo"):
                avisos.insert(0, "Se usó el plan de respaldo: el modelo no devolvió una ficha o plan válidos.")

            # Registro de la corrida (evidencia para el informe técnico).
            with open(os.path.join(CARPETA_SALIDA, "ultima_corrida.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "idea": idea_usuario.strip(),
                    "genero": genero_usuario,
                    "tiempos_s": metricas,
                    "duraciones_voz_s": duraciones_voz,
                    "avisos_texto": avisos,
                }, f, ensure_ascii=False, indent=2)

            st.session_state.html_libro = html_render
            st.session_state.metricas = metricas
            st.session_state.avisos = avisos

            contenedor_progreso.empty()
            barra.empty()

        except Exception as e:
            contenedor_progreso.empty()
            barra.empty()
            st.error(f"Ocurrió un inconveniente durante la generación: {e}. Intenta nuevamente.")

if st.session_state.html_libro:
    components.html(st.session_state.html_libro, height=760, scrolling=False)

    if st.session_state.metricas:
        m = st.session_state.metricas

        def porcentaje(valor):
            return round(100 * valor / m["total"]) if m["total"] else 0

        st.caption(
            f"Métricas de ejecución local · Total: {m['total']}s "
            f"(Texto: {m['texto']}s [{porcentaje(m['texto'])}%] · "
            f"Imágenes: {m['imagen']}s [{porcentaje(m['imagen'])}%] · "
            f"Voz: {m['voz']}s [{porcentaje(m['voz'])}%] · "
            f"Sonido: {m['sonido']}s [{porcentaje(m['sonido'])}%]) · "
            f"Extensión narrativa: {m['palabras']} palabras."
        )

    if st.session_state.avisos:
        with st.expander("Avisos de calidad del texto"):
            for aviso in st.session_state.avisos:
                st.write("• " + aviso)