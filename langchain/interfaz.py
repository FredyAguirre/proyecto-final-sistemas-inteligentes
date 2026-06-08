# -*- coding: utf-8 -*-
"""
Interfaz gráfica (web local) para el agente LangChain de la Mesa de Ayuda UNAL.

Levanta una pequeña aplicación web con Gradio donde se puede:
- escribir un ticket propio o usar ejemplos pre-cargados,
- ver la respuesta clasificada (categoría, prioridad, área, respuesta),
- desplegar el log de razonamiento del agente (Cadena de Pensamiento),
- mostrar, con botones, los fragmentos de código que se narran en el video
  (para grabar todo dentro de la misma ventana).

Uso:
    python interfaz.py
Luego abre el enlace local que aparece (http://127.0.0.1:7860).
"""
import os
import re

import gradio as gr

from agente import MODELO_LLM, procesar_ticket

COLOR_PRIORIDAD = {"alta": "#d93025", "media": "#f29900", "baja": "#188038"}

EJEMPLOS = [
    ["No puedo inscribir Machine Learning porque no hay cupos y mañana cierra la matrícula, es urgente."],
    ["Quiero hacer un reclamo de la calificación del parcial de Bases de Datos."],
    ["No puedo iniciar sesión en el SIA, me sale un error 500 al entrar al portal."],
    ["¿Cuál es el horario de atención de la biblioteca el fin de semana?"],
    ["Necesito un certificado de notas urgente para una beca que cierra hoy."],
]

# --------------------------------------------------------------------------- #
# Extracción de fragmentos de código reales (para mostrarlos en el video)
# --------------------------------------------------------------------------- #
# Lee los archivos fuente y recorta las porciones que se nombran en el guion,
# así lo que se muestra en pantalla es SIEMPRE el código verdadero del proyecto.
_DIR = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_DIR)

_SEPARADOR = re.compile(r"^\s*#\s*-+\s*#?\s*$")


def _leer(ruta_rel):
    with open(os.path.join(_RAIZ, ruta_rel), encoding="utf-8") as f:
        return f.readlines()


def _rango(lineas, ini_anchor, fin_anchor=None, ini_off=0, fin_off=0):
    """Devuelve el texto entre la línea que contiene ini_anchor y la de fin_anchor."""
    ini = next(i for i, l in enumerate(lineas) if ini_anchor in l) + ini_off
    if fin_anchor is None:
        fin = len(lineas)
    else:
        fin = next(i for i, l in enumerate(lineas) if fin_anchor in l and i > ini) + fin_off
    bloque = lineas[ini:fin]
    # Quita líneas en blanco o separadores "# ----" al final, que se ven feos.
    while bloque and (not bloque[-1].strip() or _SEPARADOR.match(bloque[-1])):
        bloque.pop()
    return "".join(bloque).rstrip("\n")


def _snippet(ruta_rel, rangos):
    lineas = _leer(ruta_rel)
    partes = [_rango(lineas, *r) for r in rangos]
    return "\n\n# ...\n\n".join(partes)


# Fragmentos en el MISMO ORDEN en que se nombran en el guion del video.
FRAGMENTOS = {
    # [0:40 - 1:40] Arquitectura del agente (4 piezas)
    "① LLM local (ChatOllama)": _snippet(
        "langchain/agente.py",
        [("# Modelo local de Ollama", "# Esquema de salida estructurada", 0, -1)],
    ),
    "② Prompt (ChatPromptTemplate)": _snippet(
        "langchain/agente.py",
        [("SYSTEM = (", "def _formatear_log", 0, 0)],
    ),
    "③ Las 2 herramientas (Tools)": _snippet(
        "langchain/herramientas.py",
        [("@tool", None, 0, 0)],
    ),
    "④ El agente (create_agent)": _snippet(
        "langchain/agente.py",
        [("def construir_agente", "def procesar_ticket", 0, 0)],
    ),
    # [1:40 - 2:30] Las herramientas y el RAG (detalle)
    "⑤ Tool 1: Machine Learning": _snippet(
        "langchain/herramientas.py",
        [("def clasificar_ticket_ml", "def buscar_base_conocimiento", -1, -1)],
    ),
    "⑥ Tool 2: RAG + base de conocimiento": _snippet(
        "langchain/herramientas.py",
        [
            ("def _construir_retriever", "def _obtener_retriever", 0, 0),
            ("def buscar_base_conocimiento", "HERRAMIENTAS =", -1, 0),
        ],
    ),
    # [2:30 - 3:00] La técnica de racionalidad: el modelo ML (Nivel 3)
    "⑦ Entrenamiento del modelo ML": _snippet(
        "ml_clasificador/entrenar.py",
        [
            ("def construir_pipeline", "def main", 0, 0),
            ("candidatos = {", "resultados = {}", 0, 0),
        ],
    ),
}

_HINT = "# 👇 Haz clic en un botón de abajo para mostrar aquí el código que estás narrando."

# Imágenes de métricas generadas por ml_clasificador/entrenar.py y la evaluación del agente.
IMAGENES = {
    "📊 Matriz de confusión": os.path.join(_RAIZ, "ml_clasificador", "matriz_confusion.png"),
    "📈 Comparación de modelos": os.path.join(_RAIZ, "ml_clasificador", "comparacion_modelos.png"),
    "🆚 Con ML vs. sin ML (Nivel 3)": os.path.join(_RAIZ, "docs", "nivel3_comparacion.png"),
}


def clasificar(mensaje):
    if not mensaje or not mensaje.strip():
        return "<p style='color:#d93025'>Escribe un mensaje primero.</p>", ""
    salida = procesar_ticket(mensaje)
    c = salida["clasificacion"]
    col = COLOR_PRIORIDAD.get(c["prioridad"], "#5f6368")
    html = f"""
    <div style="font-family:Segoe UI,Arial,sans-serif;border:1px solid #e3e3e3;border-radius:12px;
                padding:20px;background:#ffffff !important;color:#202124 !important;
                box-shadow:0 1px 4px rgba(0,0,0,.06)">
      <div style="margin-bottom:14px">
        <span style="background:#1a73e8;color:#ffffff !important;padding:5px 14px;border-radius:20px;
                     font-weight:600;font-size:14px">{c['categoria'].upper()}</span>
        <span style="background:{col};color:#ffffff !important;padding:5px 14px;border-radius:20px;
                     font-weight:600;font-size:14px;margin-left:8px">PRIORIDAD {c['prioridad'].upper()}</span>
      </div>
      <p style="margin:8px 0;color:#202124 !important">
         <b style="color:#202124 !important">🏛️ Área responsable:</b>
         <span style="color:#202124 !important">{c['area_responsable']}</span></p>
      <p style="margin:14px 0 6px;color:#202124 !important">
         <b style="color:#202124 !important">✉️ Respuesta al estudiante:</b></p>
      <div style="background:#f1f3f4;border-left:4px solid #1a73e8;padding:14px;border-radius:6px;
                  white-space:pre-wrap;line-height:1.5;color:#202124 !important">{c['respuesta_sugerida']}</div>
      <p style="margin:14px 0 4px;font-size:14px;color:#202124 !important">
         <b style="color:#202124 !important">🧠 Razonamiento:</b>
         <span style="color:#5f6368 !important">{c['razonamiento']}</span></p>
    </div>
    """
    return html, salida["log_razonamiento"]


with gr.Blocks(title="Mesa de Ayuda Inteligente UNAL") as demo:
    gr.Markdown(
        "# 🎓 Mesa de Ayuda Inteligente UNAL\n"
        "### Agente LangChain — clasificación de tickets con IA local + Machine Learning\n"
        f"*Modelo local: {MODELO_LLM} · Sede Manizales*"
    )
    with gr.Row():
        with gr.Column(scale=1):
            entrada = gr.Textbox(
                label="Mensaje del estudiante",
                placeholder="Escribe aquí la solicitud o problema del estudiante...",
                lines=5,
            )
            boton = gr.Button("Clasificar ticket", variant="primary", size="lg")
            gr.Examples(examples=EJEMPLOS, inputs=entrada,
                        label="Ejemplos (haz clic para cargar uno)")
        with gr.Column(scale=1):
            salida_html = gr.HTML(label="Resultado")
            with gr.Accordion("🔎 Ver razonamiento del agente (Cadena de Pensamiento)", open=False):
                salida_log = gr.Textbox(label="Log paso a paso (CoT)", lines=10)

    boton.click(clasificar, inputs=entrada, outputs=[salida_html, salida_log])
    entrada.submit(clasificar, inputs=entrada, outputs=[salida_html, salida_log])

    # ----------------------------------------------------------------------- #
    # Visor de código para el video: botones en el orden del guion.
    # ----------------------------------------------------------------------- #
    gr.Markdown("---\n## 🧩 Código del proyecto (para el video)\n"
                "Haz clic en cada botón **en orden** para mostrar, en el recuadro de abajo, "
                "el fragmento de código que estás narrando.")
    codigo_box = gr.Code(label="Fragmento de código", language="python", value=_HINT, lines=22)

    gr.Markdown("**[0:40 – 1:40] Arquitectura del agente** (4 piezas)")
    with gr.Row():
        for _clave in ["① LLM local (ChatOllama)", "② Prompt (ChatPromptTemplate)",
                       "③ Las 2 herramientas (Tools)", "④ El agente (create_agent)"]:
            gr.Button(_clave, size="sm").click(
                lambda c=_clave: FRAGMENTOS[c], outputs=codigo_box)

    gr.Markdown("**[1:40 – 2:30] Las herramientas y el RAG** (detalle de cada Tool)")
    with gr.Row():
        for _clave in ["⑤ Tool 1: Machine Learning", "⑥ Tool 2: RAG + base de conocimiento"]:
            gr.Button(_clave, size="sm").click(
                lambda c=_clave: FRAGMENTOS[c], outputs=codigo_box)

    gr.Markdown("**[2:30 – 3:00] La técnica de racionalidad: el modelo ML (Nivel 3)**")
    with gr.Row():
        for _clave in ["⑦ Entrenamiento del modelo ML"]:
            gr.Button(_clave, size="sm").click(
                lambda c=_clave: FRAGMENTOS[c], outputs=codigo_box)

    # Visor de imágenes (clic para mostrar la gráfica en pantalla).
    gr.Markdown("**Gráficas y resultados** (clic para mostrar la imagen)")
    imagen_box = gr.Image(label="Imagen", show_label=True, height=420)
    with gr.Row():
        for _clave in ["📊 Matriz de confusión", "📈 Comparación de modelos"]:
            gr.Button(_clave, size="sm").click(
                lambda c=_clave: IMAGENES[c], outputs=imagen_box)

    gr.Markdown("**[5:30 – 7:00] Análisis cuantitativo: ¿qué aporta el ML?**")
    with gr.Row():
        for _clave in ["🆚 Con ML vs. sin ML (Nivel 3)"]:
            gr.Button(_clave, size="sm").click(
                lambda c=_clave: IMAGENES[c], outputs=imagen_box)


if __name__ == "__main__":
    # allowed_paths: permite que Gradio sirva las imágenes de métricas, que están
    # fuera de la carpeta langchain/ (en ../ml_clasificador/).
    demo.launch(inbrowser=True, allowed_paths=[_RAIZ],
                theme=gr.themes.Soft(primary_hue="blue"))
