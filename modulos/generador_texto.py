"""Storybook: generador de texto. Cuentos de cinco escenas fieles a la premisa, 100 % local.

Estrategia de coherencia (útil para el informe técnico):
1. Ficha del cuento en pasos pequeños: ficha (personajes, lugar, objetivo, obstáculo, final)
   -> plan de 5 escenas -> descripciones visuales en inglés.
2. Memoria de hechos: tras cada escena se extraen hechos concretos y se pasan a la siguiente.
3. Cada escena se escribe viendo la escena anterior COMPLETA + hechos acumulados + ficha.
4. Validación automática con reintentos (idioma, protagonista, longitud, repetición, formato).
"""
import gc
import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, 'modelos', 'qwen3_4b')
ACTOS = ['Inicio', 'Desarrollo', 'Conflicto', 'Clímax', 'Desenlace']
MIN_PALABRAS, MAX_PALABRAS, OBJETIVO_PALABRAS = 70, 125, 95

# Muestreo recomendado para Qwen3 sin "thinking": más temperatura que antes y penalización
# de repetición suave (1.12 castigaba artículos y nombres propios y empeoraba el español).
MUESTREO = {'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05}
TEMP_NARRACION = 0.7
TEMP_JSON = 0.3
ESTADISTICAS = {'llamadas': 0, 'tokens': 0, 'segundos': 0.0}

CJK = re.compile(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]')
INGLES = re.compile(r'\b(the|and|with|was|were|his|they|their)\b', re.I)
FORMATO = re.compile(r'[\[\]{}<>*#_]')
ES_ESPANOL = re.compile(r'\b(el|la|los|las|del|de|y|con|su|sus|se|en|un|una|que|por|para|como|pero)\b', re.I)

ESTILOS = {
    'terror': 'Atmósfera de tensión creciente con detalles sensoriales inquietantes (sonidos, sombras, frío), sin gore explícito.',
    'fantasia': 'Tono maravilloso, imágenes ricas y una magia con reglas claras.',
    'misterio': 'Pistas concretas, preguntas que se acumulan y revelaciones graduales.',
    'aventura': 'Ritmo ágil, acción física y obstáculos claros.',
    'ciencia ficcion': 'Tecnología plausible, descripciones precisas y sensación de descubrimiento.',
    'infantil': 'Vocabulario sencillo, frases cortas, tono cálido y final esperanzador.',
}


# ----------------------------------------------------------------------------- utilidades
def normalizar(texto):
    texto = unicodedata.normalize('NFKD', str(texto).lower())
    return ' '.join(re.findall(r'[a-z0-9]+', ''.join(c for c in texto if not unicodedata.combining(c))))


def palabras(texto):
    return len(texto.split())


def estilo_genero(genero):
    g = normalizar(genero)
    for clave, texto in ESTILOS.items():
        if clave in g:
            return texto
    return f'Tono propio del género «{genero}».'


def limpiar(texto):
    texto = re.sub(r'<think>.*?</think>', '', str(texto), flags=re.S | re.I)
    texto = re.sub(r'```(?:json|text|markdown)?|```', '', texto, flags=re.I)
    texto = re.sub(r'^\s*(?:escena|acto|capítulo)\s*\d+\s*[:.\-–]*\s*', '', texto, flags=re.I)
    texto = re.sub(r'^\s*(?:texto|narración)\s*:\s*', '', texto, flags=re.I)
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'\s+([,.;:!?])', r'\1', texto)
    texto = re.sub(r'([.!?])(?=[A-ZÁÉÍÓÚÑ])', r'\1 ', texto)
    return texto.strip().strip('"')


def extraer_json(texto):
    decoder = json.JSONDecoder()
    texto = re.sub(r'<think>.*?</think>', '', texto, flags=re.S)
    for i, char in enumerate(texto):
        if char == '{':
            try:
                valor, _ = decoder.raw_decode(texto[i:])
                if isinstance(valor, dict):
                    return valor
            except json.JSONDecodeError:
                pass
    return {}


# ----------------------------------------------------------------------------- modelo
def cargar_modelo():
    if not os.path.isdir(RUTA_MODELO):
        raise FileNotFoundError(f'No existe el modelo local: {RUTA_MODELO}')
    if not torch.cuda.is_available():
        raise RuntimeError('Se requiere GPU NVIDIA con CUDA.')
    tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO, local_files_only=True)
    cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    modelo = AutoModelForCausalLM.from_pretrained(
        RUTA_MODELO, quantization_config=cuantizacion, device_map='auto',
        dtype=torch.float16, local_files_only=True,
    )
    modelo.eval()
    return modelo, tokenizer


def consultar(modelo, tokenizer, sistema, usuario, max_tokens=400,
              temperatura=TEMP_NARRACION, prefijo='', limpiar_salida=True):
    """prefijo: texto con el que arranca la respuesta del modelo (p. ej. '{' para forzar JSON)."""
    mensajes = [{'role': 'system', 'content': sistema}, {'role': 'user', 'content': usuario}]
    try:
        prompt = tokenizer.apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
    prompt += prefijo
    entradas = tokenizer(prompt, return_tensors='pt').to(next(modelo.parameters()).device)
    t_gen = time.perf_counter()
    with torch.inference_mode():
        salida = modelo.generate(
            **entradas, max_new_tokens=max_tokens, do_sample=True, temperature=temperatura,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, **MUESTREO,
        )
    nuevos = salida[0, entradas['input_ids'].shape[1]:]
    ESTADISTICAS['llamadas'] += 1
    ESTADISTICAS['tokens'] += int(nuevos.shape[0])
    ESTADISTICAS['segundos'] += time.perf_counter() - t_gen
    texto = prefijo + tokenizer.decode(nuevos, skip_special_tokens=True)
    return limpiar(texto) if limpiar_salida else texto


def consultar_json(modelo, tokenizer, sistema, usuario, validar, prefijo='{', max_tokens=700, intentos=3):
    """Pide JSON, lo valida y reintenta diciéndole al modelo QUÉ falló. Devuelve None si no lo logra."""
    error = ''
    for i in range(intentos):
        pedido = usuario if not error else (
            f'{usuario}\nTU RESPUESTA ANTERIOR FALLÓ: {error}. Corrígelo y devuelve solo el JSON.')
        crudo = consultar(modelo, tokenizer, sistema, pedido, max_tokens=max_tokens,
                          temperatura=TEMP_JSON if i == 0 else 0.5,
                          prefijo=prefijo, limpiar_salida=False)
        datos = extraer_json(crudo)
        if not datos:
            error = 'no era un JSON válido'
        elif CJK.search(json.dumps(datos, ensure_ascii=False)):
            error = 'contiene caracteres que no son español ni inglés'
        else:
            error = validar(datos)
            if not error:
                return datos
        print(f'[JSON] intento {i + 1}/{intentos}: {error}')
    return None


# ----------------------------------------------------------------------------- ficha del cuento
def validar_ficha(d):
    p = d.get('protagonista')
    if not isinstance(p, dict) or not str(p.get('nombre', '')).strip() or not str(p.get('tipo', '')).strip():
        return 'protagonista debe ser un objeto con nombre y tipo'
    for clave in ('titulo', 'lugar', 'tono', 'objetivo', 'obstaculo', 'final'):
        if not str(d.get(clave, '')).strip():
            return f'falta «{clave}»'
    return ''


def validar_plan(d):
    plan = d.get('plan')
    if not isinstance(plan, list) or len(plan) != 5:
        return 'plan debe tener exactamente 5 elementos'
    for p in plan:
        if not isinstance(p, dict) or palabras(str(p.get('que_sucede', ''))) < 6 or not str(p.get('que_cambia', '')).strip():
            return 'cada elemento de plan necesita que_sucede (frase completa) y que_cambia'
    return ''


def validar_visual(d):
    for clave in ('personaje_en', 'escenario_en'):
        valor = str(d.get(clave, '')).strip()
        if not valor or not valor.isascii() or ES_ESPANOL.search(valor):
            return f'«{clave}» debe estar en inglés'
    return ''


def ficha_respaldo(idea):
    return {
        'titulo': 'Una aventura extraordinaria',
        'protagonista': {'nombre': 'el protagonista', 'tipo': idea, 'oficio': '', 'rasgo': ''},
        'secundarios': [], 'lugar': 'el escenario de la premisa', 'tono': '',
        'objetivo': idea, 'obstaculo': 'una dificultad concreta', 'final': 'el objetivo se cumple',
        '_respaldo': True,
    }


def plan_respaldo():
    return [
        {'que_sucede': 'Presenta al protagonista exacto de la premisa y su objetivo; comienza la aventura.', 'que_cambia': 'Encuentra una primera pista.'},
        {'que_sucede': 'Sigue la pista y avanza hacia el objetivo original.', 'que_cambia': 'Descubre una dificultad concreta.'},
        {'que_sucede': 'La dificultad complica el objetivo y debe tomar una decisión importante.', 'que_cambia': 'Encuentra una oportunidad para resolverla.'},
        {'que_sucede': 'Enfrenta el obstáculo decisivo relacionado con el objetivo.', 'que_cambia': 'Consigue la clave para resolver la búsqueda.'},
        {'que_sucede': 'Resuelve el objetivo expresado en la premisa y muestra sus consecuencias.', 'que_cambia': 'Cierre completo y satisfactorio.'},
    ]


def construir_biblia(modelo, tokenizer, idea, genero):
    sistema = ('Eres guionista de cuentos ilustrados. Responde SOLO con un objeto JSON válido, en español. '
               'Respeta literalmente la especie, el oficio, la identidad y el objetivo de la premisa. '
               'No cambies el tipo de historia ni agregues misiones de rescate que la premisa no pide.')

    # Paso 1: ficha del cuento
    usuario = f'''PREMISA OBLIGATORIA: {idea}
GÉNERO: {genero}
Crea la ficha del cuento. Inventa un nombre propio para el protagonista (si es un animal con oficio, sigue siendo ese animal),
hasta dos personajes secundarios con nombre, un lugar concreto, el tono, el objetivo central (tomado de la premisa),
el obstáculo principal y cómo termina resolviéndose el objetivo.
Esquema exacto:
{{"titulo":"...","protagonista":{{"nombre":"...","tipo":"especie o clase exacta","oficio":"...","rasgo":"rasgo distintivo"}},"secundarios":[{{"nombre":"...","rol":"..."}}],"lugar":"...","tono":"...","objetivo":"...","obstaculo":"...","final":"..."}}'''
    ficha = consultar_json(modelo, tokenizer, sistema, usuario, validar_ficha, max_tokens=500)
    if ficha is None:
        print('[STORYBOOK] AVISO: ficha de respaldo (el modelo no devolvió una ficha válida)')
        ficha = ficha_respaldo(idea)

    # Paso 2: plan de cinco escenas
    usuario = f'''PREMISA OBLIGATORIA: {idea}
GÉNERO: {genero}
FICHA:
{ficha_texto(ficha)}
Diseña exactamente cinco escenas con progresión causal: 1) presenta al protagonista y su lugar; 2) primer paso hacia el objetivo;
3) el obstáculo complica las cosas; 4) enfrentamiento decisivo; 5) resuelve el objetivo tal como indica «final».
Cada escena debe tener una acción distinta y concreta. Usa solo personajes de la ficha.
Esquema exacto:
{{"plan":[{{"que_sucede":"acción concreta","que_cambia":"consecuencia"}}, ... cinco elementos ...]}}'''
    plan = consultar_json(modelo, tokenizer, sistema, usuario, validar_plan, prefijo='{"plan":', max_tokens=700)
    if plan is None:
        print('[STORYBOOK] AVISO: plan de respaldo (el modelo no devolvió un plan válido)')
        ficha['_respaldo'] = True
        ficha['plan'] = plan_respaldo()
    else:
        ficha['plan'] = plan['plan']

    # Paso 3: descripciones visuales en inglés (para el generador de imágenes)
    sistema_en = 'Return ONLY a valid JSON object with keys "personaje_en" and "escenario_en". Both values in English.'
    usuario_en = f'''Protagonist: {ficha_texto(ficha)}
"personaje_en": visual description of the protagonist (species, clothing, colors, distinctive features), 20-40 words.
"escenario_en": visual description of the main setting and lighting, 15-30 words.'''
    visual = consultar_json(modelo, tokenizer, sistema_en, usuario_en, validar_visual, max_tokens=250, intentos=2)
    if visual:
        ficha['personaje_en'] = str(visual['personaje_en']).strip()
        ficha['escenario_en'] = str(visual['escenario_en']).strip()
    return ficha


def ficha_texto(b):
    p = b.get('protagonista') or {}
    if not isinstance(p, dict):
        p = {'nombre': str(p), 'tipo': ''}
    secundarios = '; '.join(
        f"{s.get('nombre', '')} ({s.get('rol', '')})" for s in b.get('secundarios', []) if isinstance(s, dict)
    ) or 'ninguno'
    return (f"Título: {b.get('titulo', '')}\n"
            f"Protagonista: {p.get('nombre', '')} — {p.get('tipo', '')}. Oficio: {p.get('oficio', '') or 'sin especificar'}. "
            f"Rasgo: {p.get('rasgo', '') or 'sin especificar'}\n"
            f"Secundarios: {secundarios}\n"
            f"Lugar: {b.get('lugar', '')}\nTono: {b.get('tono', '')}\n"
            f"Objetivo: {b.get('objetivo', '')}\nObstáculo: {b.get('obstaculo', '')}\nFinal previsto: {b.get('final', '')}")


def claves_protagonista(biblia):
    p = biblia.get('protagonista') or {}
    if not isinstance(p, dict):
        p = {'nombre': str(p), 'tipo': ''}
    return [t for t in normalizar(f"{p.get('nombre', '')} {p.get('tipo', '')}").split() if len(t) >= 4]


# ----------------------------------------------------------------------------- escenas
def problemas_escena(texto, anteriores, claves):
    problemas = []
    n = palabras(texto)
    if n < MIN_PALABRAS:
        problemas.append('demasiado breve')
    if n > MAX_PALABRAS:
        problemas.append('demasiado extenso')
    if not re.search(r'[.!?]["»]?$', texto.strip()):
        problemas.append('termina sin cerrar la oración')
    if CJK.search(texto):
        problemas.append('contiene caracteres que no son español')
    if len(INGLES.findall(texto)) >= 3:
        problemas.append('contiene frases en inglés')
    if FORMATO.search(texto):
        problemas.append('contiene símbolos de formato')
    norm = normalizar(texto)
    if claves and not any(c in norm for c in claves):
        problemas.append('no menciona al protagonista')
    for anterior in anteriores:
        if SequenceMatcher(None, norm, normalizar(anterior['texto'])).ratio() > 0.6:
            problemas.append('repite demasiado una escena anterior')
            break
    if anteriores and norm.split()[:5] == normalizar(anteriores[-1]['texto']).split()[:5]:
        problemas.append('empieza igual que la escena anterior')
    return problemas


def oraciones(texto):
    return [o.strip() for o in re.split(r'(?<=[.!?…])\s+', texto) if o.strip()]


def reparar_escena(texto, anteriores):
    """Quita oraciones copiadas de escenas previas, descarta una última oración cortada y limita la longitud."""
    previas = [normalizar(o) for e in anteriores for o in oraciones(e['texto'])]
    nuevas = [o for o in oraciones(texto)
              if not any(SequenceMatcher(None, normalizar(o), p).ratio() > 0.8 for p in previas)]
    if len(nuevas) > 1 and not re.search(r'[.!?…]["»]?$', nuevas[-1]):
        nuevas = nuevas[:-1]
    salida = []
    for o in nuevas:
        if salida and palabras(' '.join(salida + [o])) > MAX_PALABRAS:
            break
        salida.append(o)
    return ' '.join(salida) if salida else texto


def redactar_escena(modelo, tokenizer, numero, biblia, idea, genero, anteriores):
    paso = biblia['plan'][numero - 1]
    claves = claves_protagonista(biblia)
    memoria = '\n'.join(f"Escena {e['numero']}: {' '.join(e.get('hechos', []))}" for e in anteriores) or 'Ninguna (es la primera escena).'
    anterior = (' '.join(oraciones(anteriores[-1]['texto'])[-2:]) if anteriores
                else 'No hay; esta es la primera escena.')
    sistema = ('Eres un escritor de cuentos en español neutro. Escribes en tercera persona y en pasado, con frases de longitud '
               'variada, verbos concretos y vocabulario natural, sin anglicismos ni traducciones literales. '
               'Devuelves SOLO el párrafo narrativo, sin título, sin encabezados y sin explicaciones. '
               'Escribes únicamente en español.')
    usuario = f'''IDEA ORIGINAL (prioridad máxima): {idea}
GÉNERO: {genero}. {estilo_genero(genero)}

FICHA INALTERABLE:
{ficha_texto(biblia)}

HECHOS YA ESTABLECIDOS (no los contradigas):
{memoria}

FINAL DE LA ESCENA ANTERIOR (solo para dar continuidad; NO lo copies):
{anterior}

AHORA ESCRIBE SOLO LA ESCENA {numero} DE 5 ({ACTOS[numero - 1]}).
Acción obligatoria: {paso.get('que_sucede', '')}
Consecuencia: {paso.get('que_cambia', '')}

REGLAS:
- Entre 80 y 110 palabras, en un solo párrafo. Escribe SOLO la escena nueva: no repitas ni resumas las anteriores.
- {'Empieza presentando al protagonista y su lugar.' if numero == 1 else 'Retoma el final de la escena anterior sin repetir lo ya contado.'}
- Usa los nombres de la ficha; no inventes personajes importantes nuevos ni cambies al protagonista.
- No repitas descripciones ni frases de escenas previas; evita muletillas como «de repente» o «sin embargo» más de una vez.
- {'CIERRA LA HISTORIA resolviendo explícitamente el objetivo original.' if numero == 5 else 'No resuelvas todavía el objetivo principal.'}'''
    mejor, mejor_problemas = '', ['sin candidato']
    for intento in range(3):
        peticion = usuario if intento == 0 or not mejor_problemas else (
            f"{usuario}\nCORRIGE ESTOS PROBLEMAS de tu versión anterior: {', '.join(mejor_problemas)}.")
        candidato = reparar_escena(
            consultar(modelo, tokenizer, sistema, peticion, max_tokens=280,
                      temperatura=TEMP_NARRACION - 0.05 * intento), anteriores)
        problemas = problemas_escena(candidato, anteriores, claves)
        print(f'[ESCENA {numero}/5] intento {intento + 1}: {palabras(candidato)} palabras | {problemas or "OK"}')
        if (len(problemas), abs(palabras(candidato) - OBJETIVO_PALABRAS)) < (
                len(mejor_problemas) if mejor else 99, abs(palabras(mejor) - OBJETIVO_PALABRAS) if mejor else 999):
            mejor, mejor_problemas = candidato, problemas
        if not problemas:
            return candidato, []
    return mejor, mejor_problemas


def validar_hechos(d):
    if not isinstance(d.get('hechos'), list) or not d['hechos']:
        return '«hechos» debe ser una lista no vacía'
    return ''


def es_ingles(valor):
    valor = str(valor or '').strip()
    return bool(valor) and valor.isascii() and not ES_ESPANOL.search(valor)


def validar_visual_escena(d):
    for clave in ('accion_en', 'lugar_en'):
        if not es_ingles(d.get(clave)):
            return f'«{clave}» debe estar escrito en inglés, no en español'
    return ''


def analizar_escena(modelo, tokenizer, escena, biblia):
    """Dos llamadas separadas: hechos en español (memoria) y descripción visual SOLO en inglés."""
    sistema_es = 'Devuelve SOLO un objeto JSON válido con la clave "hechos".'
    usuario_es = f'''Texto de la escena {escena['numero']}:
{escena['texto']}

"hechos": lista de 2 o 3 frases cortas en español (máximo 15 palabras cada una) con hechos concretos de ESTA escena: nombres, lugares, objetos encontrados o perdidos, heridas, promesas, decisiones.'''
    datos = consultar_json(modelo, tokenizer, sistema_es, usuario_es, validar_hechos,
                           prefijo='{"hechos":', max_tokens=200, intentos=2) or {}
    hechos = [str(h).strip() for h in (datos.get('hechos') or [])[:3] if str(h).strip()]
    if not hechos:
        hechos = [re.split(r'(?<=[.!?])\s', escena['texto'])[0]]

    sistema_en = ('You write prompts for an image generator. Return ONLY a valid JSON object. '
                  'Every value MUST be written in English, even though the story is in Spanish.')
    usuario_en = f'''Story scene (in Spanish): {escena['texto']}
Protagonist look: {biblia.get('personaje_en', '')}

Return a JSON object with these keys, all values in English:
"accion_en": ONE visible action the protagonist performs in this scene, 8-16 words; do not use the character's name, say "the character".
"lugar_en": the specific place where this scene happens plus lighting or time of day, 4-10 words.
"sonido_en": plausible ambient sound for this scene, 4-10 words, no voices.'''
    visual = consultar_json(modelo, tokenizer, sistema_en, usuario_en, validar_visual_escena,
                            prefijo='{"accion_en":"', max_tokens=150, intentos=3) or {}
    accion = str(visual.get('accion_en') or '').strip(' ,.;')
    lugar = str(visual.get('lugar_en') or '').strip(' ,.;')
    sonido = str(visual.get('sonido_en') or '').strip(' ,.;')
    if not es_ingles(accion):
        print(f'[TEXTO] AVISO escena {escena["numero"]}: sin acción visual válida en inglés')
        accion = f'main character performing the key action of scene {escena["numero"]}'
    if not es_ingles(lugar):
        lugar = ''  # el generador de imágenes usará el escenario global
    if not es_ingles(sonido):
        sonido = 'subtle environmental ambience and atmospheric wind'
    return hechos, accion, lugar, sonido


# ----------------------------------------------------------------------------- API pública
def generar_historia(idea: str, genero: str, callback_progreso=None) -> dict:
    if not isinstance(idea, str) or not idea.strip():
        raise ValueError('Debes proporcionar una idea.')
    if not isinstance(genero, str) or not genero.strip():
        raise ValueError('Debes seleccionar un género.')
    idea, genero = idea.strip(), genero.strip()
    inicio = time.perf_counter()
    ESTADISTICAS.update(llamadas=0, tokens=0, segundos=0.0)
    modelo = tokenizer = None
    try:
        modelo, tokenizer = cargar_modelo()
        t0 = time.perf_counter()
        biblia = construir_biblia(modelo, tokenizer, idea, genero)
        t_biblia = time.perf_counter() - t0
        print('[STORYBOOK] Título:', biblia.get('titulo'))
        print('[STORYBOOK] Objetivo:', biblia.get('objetivo'))
        escenas, avisos = [], []
        for numero, acto in enumerate(ACTOS, 1):
            texto, problemas = redactar_escena(modelo, tokenizer, numero, biblia, idea, genero, escenas)
            escena = {'numero': numero, 'acto': acto, 'texto': texto,
                      'hechos': [], 'accion_en': '', 'lugar_en': '', 'sonido_en': ''}
            (escena['hechos'], escena['accion_en'],
             escena['lugar_en'], escena['sonido_en']) = analizar_escena(
                modelo, tokenizer, escena, biblia)
            if problemas:
                avisos.append(f'Escena {numero}: {", ".join(problemas)}')
            escenas.append(escena)
            if callback_progreso:
                callback_progreso(numero, 5)
        e = ESTADISTICAS
        if e['segundos']:
            print(f"[TEXTO] {e['llamadas']} llamadas | {e['tokens']} tokens | "
                  f"{e['segundos']:.0f}s generando | {e['tokens'] / e['segundos']:.1f} tokens/s")
        return {
            'titulo': biblia.get('titulo') or 'Relato ilustrado',
            'personaje_en': biblia.get('personaje_en') or 'the main character described in the story premise',
            'escenario_en': biblia.get('escenario_en') or 'illustrated storybook setting',
            'escenas': escenas,
            'biblia': biblia,
            'avisos': avisos,
            'uso_respaldo': bool(biblia.get('_respaldo')),
            'tiempo_biblia_s': round(t_biblia, 2),
            'tiempo_texto_s': round(time.perf_counter() - inicio, 2),
        }
    finally:
        if modelo is not None:
            del modelo
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()