import base64
import json

def _archivo_a_base64(ruta: str, mime: str) -> str:
    with open(ruta, "rb") as f:
        datos = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{datos}"

def construir_html_libro(datos_historia: dict, genero: str, rutas_imagenes: list, rutas_voces: list, rutas_sonidos: list) -> str:
    paginas = []
    for i, escena in enumerate(datos_historia["escenas"]):
        paginas.append({
            "numero": escena["numero"],
            "acto": escena.get("acto", f"Escena {escena['numero']}"),
            "texto": escena["texto"],
            "imagen": _archivo_a_base64(rutas_imagenes[i], "image/png"),
            "voz": _archivo_a_base64(rutas_voces[i], "audio/wav"),
            "sonido": _archivo_a_base64(rutas_sonidos[i], "audio/wav")
        })

    payload_json = json.dumps({
        "titulo": datos_historia["titulo"],
        "genero": genero,
        "paginas": paginas
    }, ensure_ascii=False)

    return f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-noche: #0b101b;
    --cuero: #131c2e;
    --cuero-borde: #c9a256;
    --papel: #f5eedc;
    --papel-sombra: #e4d8bc;
    --tinta: #231d16;
    --dorado: #d4af63;
    --bosque: #1b3b32;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: transparent;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #f3efe6;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 8px 24px 8px;
    overflow-x: hidden;
  }}

  /* Escenario 3D del libro */
  .escenario-libro {{
    width: 100%;
    max-width: 1120px;
    perspective: 2400px;
    margin: 0 auto;
  }}

  /* PORTADA */
  .portada {{
    width: 100%;
    max-width: 560px;
    min-height: 620px;
    margin: 0 auto;
    background: radial-gradient(circle at 30% 20%, #1c2942 0%, #0f1624 70%, #090d16 100%);
    border: 2px solid var(--dorado);
    border-radius: 6px 18px 18px 6px;
    box-shadow:
      -12px 0 0 #0a0f1a,
      -15px 0 0 var(--dorado),
      0 25px 60px rgba(0, 0, 0, 0.75),
      inset 0 0 40px rgba(212, 175, 99, 0.12);
    padding: 48px 40px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
    text-align: center;
    position: relative;
    transition: transform 0.7s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.5s ease;
  }}

  .marco-interior {{
    border: 1px solid rgba(212, 175, 99, 0.45);
    width: 100%;
    height: 100%;
    min-height: 520px;
    padding: 36px 28px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
  }}

  .etiqueta-genero {{
    font-family: 'Cinzel', serif;
    font-size: 0.78rem;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--dorado);
    border-bottom: 1px solid rgba(212, 175, 99, 0.4);
    padding-bottom: 8px;
  }}

  .titulo-portada {{
    font-family: 'Cinzel', serif;
    font-size: clamp(1.75rem, 3vw, 2.45rem);
    line-height: 1.25;
    color: #fbf5e6;
    text-shadow: 0 2px 12px rgba(0,0,0,0.6);
    margin: 24px 0;
  }}

  .subtitulo-portada {{
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.2rem;
    color: #cbbba0;
  }}

  .btn-editorial {{
    background: linear-gradient(180deg, #dfc07c 0%, #b88e3d 100%);
    color: #14110c;
    font-family: 'Cinzel', serif;
    font-weight: 700;
    font-size: 0.92rem;
    letter-spacing: 0.1em;
    border: none;
    border-radius: 4px;
    padding: 14px 32px;
    cursor: pointer;
    box-shadow: 0 6px 20px rgba(0,0,0,0.45);
    transition: transform 0.2s ease, filter 0.2s ease;
  }}

  .btn-editorial:hover {{
    transform: translateY(-2px);
    filter: brightness(1.07);
  }}

  /* LIBRO ABIERTO */
  .libro-abierto {{
    display: none;
    width: 100%;
    background: var(--cuero);
    border: 2px solid rgba(212, 175, 99, 0.55);
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.8);
    position: relative;
  }}

  .pliego {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    background: var(--papel);
    border-radius: 8px;
    position: relative;
    min-height: 560px;
    box-shadow: inset 0 0 50px rgba(138, 114, 72, 0.25);
    transform-style: preserve-3d;
  }}

  /* Pliegue central del libro */
  .pliego::after {{
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 44px;
    transform: translateX(-50%);
    background: linear-gradient(90deg,
      rgba(0,0,0,0) 0%,
      rgba(45, 32, 16, 0.14) 42%,
      rgba(20, 14, 6, 0.32) 50%,
      rgba(45, 32, 16, 0.14) 58%,
      rgba(0,0,0,0) 100%);
    pointer-events: none;
    z-index: 5;
  }}

  .pagina {{
    padding: 32px 36px;
    color: var(--tinta);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
    background-image: radial-gradient(rgba(160, 136, 94, 0.12) 1px, transparent 0);
    background-size: 22px 22px;
    transition: opacity 0.35s ease;
  }}

  .pagina-izquierda {{
    border-right: 1px solid rgba(120, 96, 60, 0.22);
    align-items: center;
    justify-content: center;
  }}

  .marco-ilustracion {{
    width: 100%;
    max-width: 440px;
    aspect-ratio: 1 / 1;
    border: 6px solid #fffaf0;
    outline: 1px solid rgba(156, 122, 62, 0.55);
    box-shadow: 0 12px 28px rgba(35, 25, 12, 0.28);
    overflow: hidden;
    border-radius: 3px;
    background: #1a1815;
  }}

  .marco-ilustracion img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.6s ease;
  }}

  .pie-ilustracion {{
    margin-top: 14px;
    font-family: 'Cinzel', serif;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    color: #6b583c;
    text-transform: uppercase;
  }}

  .pagina-derecha {{
    justify-content: space-between;
  }}

  .cabecera-pagina {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(130, 102, 58, 0.3);
    padding-bottom: 10px;
    margin-bottom: 18px;
    font-family: 'Cinzel', serif;
    font-size: 0.82rem;
    color: #6e5633;
    letter-spacing: 0.08em;
  }}

  .texto-narrativo {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.26rem;
    line-height: 1.72;
    color: var(--tinta);
    text-align: justify;
    max-height: 395px;
    overflow-y: auto;
    padding-right: 8px;
  }}

  .texto-narrativo::first-letter {{
    font-family: 'Cinzel', serif;
    font-size: 3.1rem;
    float: left;
    line-height: 0.85;
    padding-right: 10px;
    padding-top: 4px;
    color: #7c5821;
  }}

  .folio-pagina {{
    text-align: right;
    font-family: 'Cinzel', serif;
    font-size: 0.82rem;
    color: #7a6443;
    margin-top: 14px;
  }}

  /* Hoja animada de paso de página (Page-Flip 3D) */
  .hoja-animada {{
    position: absolute;
    top: 0;
    right: 0;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, #e7dcc2 0%, #f7f1e1 18%, #efe4ca 100%);
    transform-origin: left center;
    pointer-events: none;
    opacity: 0;
    z-index: 20;
    border-radius: 0 8px 8px 0;
    box-shadow: -10px 0 30px rgba(0,0,0,0.28);
  }}

  .hoja-animada.girar-siguiente {{
    animation: flipNext 0.65s cubic-bezier(0.45, 0.05, 0.55, 0.95) forwards;
  }}

  .hoja-animada.girar-anterior {{
    animation: flipPrev 0.65s cubic-bezier(0.45, 0.05, 0.55, 0.95) forwards;
  }}

  @keyframes flipNext {{
    0%   {{ opacity: 0.95; transform: rotateY(0deg); }}
    50%  {{ opacity: 1;    transform: rotateY(-95deg); }}
    100% {{ opacity: 0;    transform: rotateY(-180deg); }}
  }}

  @keyframes flipPrev {{
    0%   {{ opacity: 0.95; transform: rotateY(-180deg); }}
    50%  {{ opacity: 1;    transform: rotateY(-85deg); }}
    100% {{ opacity: 0;    transform: rotateY(0deg); }}
  }}

  /* CONSOLA DE AUDIO DUAL Y NAVEGACIÓN */
  .barra-controles {{
    margin-top: 16px;
    background: #101726;
    border: 1px solid rgba(212, 175, 99, 0.35);
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    justify-content: space-between;
  }}

  .grupo-nav {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .btn-nav {{
    background: #1c273d;
    color: #f2ead8;
    border: 1px solid rgba(212, 175, 99, 0.45);
    border-radius: 6px;
    padding: 9px 16px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.86rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }}

  .btn-nav:hover:not(:disabled) {{
    background: #273654;
    border-color: var(--dorado);
  }}

  .btn-nav:disabled {{
    opacity: 0.38;
    cursor: not-allowed;
  }}

  .indicador-escena {{
    font-family: 'Cinzel', serif;
    font-size: 0.92rem;
    color: var(--dorado);
    padding: 0 8px;
    font-weight: 600;
  }}

  .grupo-audio {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
  }}

  .btn-play {{
    background: var(--bosque);
    color: #f4efe4;
    border: 1px solid var(--dorado);
    border-radius: 6px;
    padding: 9px 18px;
    font-weight: 600;
    font-size: 0.88rem;
    cursor: pointer;
    transition: background 0.2s ease;
  }}

  .btn-play:hover {{
    background: #244f43;
  }}

  .control-canal {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.8rem;
    color: #d6cbb5;
  }}

  .control-canal input[type="range"] {{
    width: 82px;
    accent-color: var(--dorado);
    cursor: pointer;
  }}

  .btn-mute {{
    background: transparent;
    color: var(--dorado);
    border: 1px solid rgba(212, 175, 99, 0.4);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 0.75rem;
    cursor: pointer;
  }}

  .btn-mute.silenciado {{
    opacity: 0.5;
    text-decoration: line-through;
  }}

  /* RESPONSIVE MÓVIL Y TABLET */
  @media (max-width: 860px) {{
    .pliego {{
      grid-template-columns: 1fr;
    }}
    .pliego::after, .hoja-animada {{
      display: none;
    }}
    .pagina {{
      padding: 22px 20px;
    }}
    .pagina-izquierda {{
      border-right: none;
      border-bottom: 1px solid rgba(120, 96, 60, 0.25);
    }}
    .barra-controles {{
      flex-direction: column;
      align-items: stretch;
    }}
    .grupo-nav, .grupo-audio {{
      justify-content: center;
    }}
  }}
</style>
</head>
<body>

<div class="escenario-libro">
  <!-- PORTADA -->
  <div id="vistaPortada" class="portada">
    <div class="marco-interior">
      <div class="etiqueta-genero" id="portadaGenero"></div>
      <div>
        <h1 class="titulo-portada" id="portadaTitulo"></h1>
        <p class="subtitulo-portada">Edición Ilustrada Interactivo · Cinco Escenas</p>
      </div>
      <button class="btn-editorial" onclick="abrirLibro()">Abrir el Libro</button>
    </div>
  </div>

  <!-- LIBRO ABIERTO -->
  <div id="vistaLibro" class="libro-abierto">
    <div class="pliego">
      <div class="hoja-animada" id="hojaAnimada"></div>

      <!-- PÁGINA IZQUIERDA: ILUSTRACIÓN -->
      <div class="pagina pagina-izquierda">
        <div class="marco-ilustracion">
          <img id="imgEscena" src="" alt="Ilustración de la escena">
        </div>
        <div class="pie-ilustracion" id="pieIlustracion">Lámina I</div>
      </div>

      <!-- PÁGINA DERECHA: RELATO -->
      <div class="pagina pagina-derecha">
        <div>
          <div class="cabecera-pagina">
            <span id="cabeceraTitulo"></span>
            <span id="cabeceraActo"></span>
          </div>
          <div class="texto-narrativo" id="textoEscena"></div>
        </div>
        <div class="folio-pagina" id="folioPagina">Página 1 de 5</div>
      </div>
    </div>

    <!-- CONTROLES DE NAVEGACIÓN Y AUDIO SIMULTÁNEO -->
    <div class="barra-controles">
      <div class="grupo-nav">
        <button class="btn-nav" id="btnPortada" onclick="volverPortada()">Portada</button>
        <button class="btn-nav" id="btnAnterior" onclick="cambiarPagina(-1)">Anterior</button>
        <span class="indicador-escena" id="indicadorEscena">1 de 5</span>
        <button class="btn-nav" id="btnSiguiente" onclick="cambiarPagina(1)">Siguiente</button>
      </div>

      <div class="grupo-audio">
        <button class="btn-play" id="btnPlayPausa" onclick="alternarReproduccion()">Escuchar Escena</button>

        <div class="control-canal">
          <span>Voz</span>
          <input type="range" id="volVoz" min="0" max="1" step="0.05" value="1" oninput="actualizarVolumenes()">
        </div>

        <div class="control-canal">
          <span>Ambiente</span>
          <input type="range" id="volAmbiente" min="0" max="1" step="0.05" value="0.25" oninput="actualizarVolumenes()">
          <button class="btn-mute" id="btnMuteAmbiente" onclick="alternarMuteAmbiente()">Activo</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  const HISTORIA = {payload_json};
  const ROMANOS = ["I", "II", "III", "IV", "V"];
  let indiceActual = 0;
  let reproduciendo = false;
  let ambienteSilenciado = false;

  // Elementos HTML5 Audio controlados por Web Audio API para mezcla simultánea sin desfase
  const audioVoz = new Audio();
  const audioAmbiente = new Audio();
  audioAmbiente.loop = true;

  let audioCtx = null;
  let gainVoz = null;
  let gainAmbiente = null;

  function inicializarWebAudio() {{
    if (!audioCtx) {{
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContext();
      const srcVoz = audioCtx.createMediaElementSource(audioVoz);
      const srcAmb = audioCtx.createMediaElementSource(audioAmbiente);

      gainVoz = audioCtx.createGain();
      gainAmbiente = audioCtx.createGain();

      srcVoz.connect(gainVoz).connect(audioCtx.destination);
      srcAmb.connect(gainAmbiente).connect(audioCtx.destination);
      actualizarVolumenes();
    }}
    if (audioCtx.state === "suspended") {{
      audioCtx.resume();
    }}
  }}

  function actualizarVolumenes() {{
    const vVoz = parseFloat(document.getElementById("volVoz").value);
    const vAmb = ambienteSilenciado ? 0 : parseFloat(document.getElementById("volAmbiente").value);
    if (gainVoz && gainAmbiente) {{
      gainVoz.gain.value = vVoz;
      gainAmbiente.gain.value = vAmb;
    }} else {{
      audioVoz.volume = vVoz;
      audioAmbiente.volume = vAmb;
    }}
  }}

  function alternarMuteAmbiente() {{
    ambienteSilenciado = !ambienteSilenciado;
    const btn = document.getElementById("btnMuteAmbiente");
    btn.textContent = ambienteSilenciado ? "Silenciado" : "Activo";
    btn.classList.toggle("silenciado", ambienteSilenciado);
    actualizarVolumenes();
  }}

  function detenerAudios() {{
    audioVoz.pause();
    audioVoz.currentTime = 0;
    audioAmbiente.pause();
    audioAmbiente.currentTime = 0;
    reproduciendo = false;
    document.getElementById("btnPlayPausa").textContent = "Escuchar Escena";
  }}

  audioVoz.addEventListener("ended", () => {{
    // Cuando termina la narración, atenúa suavemente y detiene la ambientación
    detenerAudios();
  }});

  function alternarReproduccion() {{
    inicializarWebAudio();
    if (!reproduciendo) {{
      actualizarVolumenes();
      Promise.all([audioVoz.play(), audioAmbiente.play()])
        .then(() => {{
          reproduciendo = true;
          document.getElementById("btnPlayPausa").textContent = "Pausar Escena";
        }})
        .catch(err => console.warn("Interacción requerida para audio:", err));
    }} else {{
      audioVoz.pause();
      audioAmbiente.pause();
      reproduciendo = false;
      document.getElementById("btnPlayPausa").textContent = "Reanudar Escena";
    }}
  }}

  function renderizarEscena(idx, direccion = 0) {{
    detenerAudios();
    const hoja = document.getElementById("hojaAnimada");
    hoja.classList.remove("girar-siguiente", "girar-anterior");
    void hoja.offsetWidth; // Reinicia animación CSS
    if (direccion === 1) hoja.classList.add("girar-siguiente");
    if (direccion === -1) hoja.classList.add("girar-anterior");

    const escena = HISTORIA.paginas[idx];
    document.getElementById("imgEscena").src = escena.imagen;
    document.getElementById("pieIlustracion").textContent = `Lámina ${{ROMANOS[idx]}} · ${{escena.acto}}`;
    document.getElementById("cabeceraTitulo").textContent = HISTORIA.titulo;
    document.getElementById("cabeceraActo").textContent = `Escena ${{escena.numero}} de 5`;
    document.getElementById("textoEscena").textContent = escena.texto;
    document.getElementById("textoEscena").scrollTop = 0;
    document.getElementById("folioPagina").textContent = `Escena ${{escena.numero}} de 5`;
    document.getElementById("indicadorEscena").textContent = `${{escena.numero}} de 5`;

    // Cargar pistas de la escena actual sin auto-reproducir hasta interacción
    audioVoz.src = escena.voz;
    audioAmbiente.src = escena.sonido;
    audioVoz.load();
    audioAmbiente.load();

    document.getElementById("btnAnterior").disabled = (idx === 0);
    document.getElementById("btnSiguiente").disabled = (idx === HISTORIA.paginas.length - 1);
  }}

  function abrirLibro() {{
    document.getElementById("vistaPortada").style.display = "none";
    document.getElementById("vistaLibro").style.display = "block";
    indiceActual = 0;
    renderizarEscena(0, 1);
  }}

  function volverPortada() {{
    detenerAudios();
    document.getElementById("vistaLibro").style.display = "none";
    document.getElementById("vistaPortada").style.display = "flex";
  }}

  function cambiarPagina(delta) {{
    const nuevo = indiceActual + delta;
    if (nuevo < 0 || nuevo >= HISTORIA.paginas.length) return;
    indiceActual = nuevo;
    renderizarEscena(indiceActual, delta);
  }}

  // Inicializar portada
  document.getElementById("portadaGenero").textContent = HISTORIA.genero;
  document.getElementById("portadaTitulo").textContent = HISTORIA.titulo;
</script>
</body>
</html>
"""