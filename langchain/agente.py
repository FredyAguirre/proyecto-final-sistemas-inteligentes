# -*- coding: utf-8 -*-
"""
NIVEL 2 — Agente LangChain de la Mesa de Ayuda UNAL.

Reimplementa en LangChain el mismo caso del flujo n8n del Nivel 1: clasificar un
ticket de estudiante (categoría + prioridad), recuperar información oficial (RAG)
y redactar una respuesta. El agente integra además el clasificador de Machine
Learning del Nivel 3 como una herramienta.

Componentes (diapositiva 07):
- LLM:                ChatOllama (modelo local, p. ej. qwen2.5:7b).
- Tools:             clasificar_ticket_ml (ML, Nivel 3) y buscar_base_conocimiento (RAG, Nivel 2).
- Agent:             create_agent (LangChain 1.x, grafo de razonamiento sobre langgraph).
- ChatPromptTemplate: estructura el paso final de salida estructurada (system + human).
- CoT:               razonamiento paso a paso vía uso de herramientas (estilo ReAct);
                     queda registrado en los mensajes del agente (log de razonamiento, diapo 08-09).
- Salida estructurada: el LLM con with_structured_output produce el JSON final validado.
"""
import os
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from herramientas import buscar_base_conocimiento, HERRAMIENTAS

# Modelo local de Ollama. qwen2.5:3b cabe completo en una GPU de 6 GB (RTX 2060),
# lo que acelera la inferencia ~5-10x frente a qwen2.5:7b (que se reparte CPU/GPU).
MODELO_LLM = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

llm = ChatOllama(model=MODELO_LLM, temperature=0.2)


# --------------------------------------------------------------------------- #
# Esquema de salida estructurada
# --------------------------------------------------------------------------- #
class Clasificacion(BaseModel):
    """Resultado del triage de un ticket de la Mesa de Ayuda UNAL."""
    categoria: Literal["matricula", "notas", "plataforma", "biblioteca", "certificados"] = Field(
        description="Categoría del ticket.")
    prioridad: Literal["alta", "media", "baja"] = Field(
        description="alta = bloqueo o fecha límite inminente; media = importante; baja = consulta general.")
    area_responsable: str = Field(description="Área de la universidad que debe atender el caso.")
    respuesta_sugerida: str = Field(description="Respuesta cordial y profesional para el estudiante.")
    razonamiento: str = Field(description="Breve explicación de por qué se eligió la categoría y la prioridad.")


# --------------------------------------------------------------------------- #
# Prompt del agente: razona paso a paso y usa las herramientas (CoT / ReAct)
# --------------------------------------------------------------------------- #
SYSTEM = (
    "Eres el asistente de la Mesa de Ayuda de la Universidad Nacional de Colombia "
    "(UNAL), Sede Manizales. Atiendes tickets de estudiantes.\n\n"
    "Razona paso a paso y sigue SIEMPRE este procedimiento:\n"
    "1. Llama a la herramienta 'clasificar_ticket_ml' para obtener la categoría probable "
    "del ticket según el modelo de Machine Learning.\n"
    "2. Llama a la herramienta 'buscar_base_conocimiento' para recuperar los procedimientos, "
    "plazos y contactos oficiales relacionados con el ticket.\n"
    "3. Con esa evidencia, determina la categoría definitiva y la prioridad "
    "(alta = bloqueo o fecha límite inminente; media = importante; baja = consulta general) "
    "y redacta una respuesta cordial firmada como 'Mesa de Ayuda UNAL Sede Manizales'.\n"
    "Apóyate en la información recuperada; no inventes datos de contacto ni procedimientos.\n"
    "Responde SIEMPRE en español."
)

# Cadena de salida estructurada (ChatPromptTemplate + LLM): convierte el análisis del
# agente en un JSON validado contra el esquema Clasificacion.
prompt_estructura = ChatPromptTemplate.from_messages([
    ("system", "A partir del ticket y del análisis recopilado por el agente (que incluye la "
               "predicción del modelo ML y la información oficial recuperada por RAG), entrega la "
               "clasificación final en el formato estructurado solicitado. "
               "TODOS los campos deben estar redactados en español."),
    ("human", "Ticket del estudiante:\n{texto}\n\nAnálisis y evidencia recopilada:\n{analisis}"),
])
cadena_estructurada = prompt_estructura | llm.with_structured_output(Clasificacion)


def _formatear_log(mensajes) -> str:
    """Extrae de los mensajes del agente un log legible de razonamiento (CoT)."""
    lineas, paso = [], 0
    for m in mensajes:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                paso += 1
                lineas.append(f"Paso {paso}: el agente decide usar '{tc['name']}' con {tc['args']}")
        elif isinstance(m, ToolMessage):
            obs = str(m.content).strip().replace("\n", " ")
            if len(obs) > 300:
                obs = obs[:300] + "..."
            lineas.append(f"   Observación de '{m.name}': {obs}")
    return "\n".join(lineas) if lineas else "(El agente no usó herramientas)"


def construir_agente(usar_ml: bool = True):
    """Crea el agente. usar_ml=False excluye la herramienta de ML (línea base Nivel 2)."""
    tools = list(HERRAMIENTAS) if usar_ml else [buscar_base_conocimiento]
    return create_agent(llm, tools, system_prompt=SYSTEM)


def procesar_ticket(texto: str, usar_ml: bool = True) -> dict:
    """Procesa un ticket y devuelve la clasificación estructurada + el log de razonamiento."""
    agente = construir_agente(usar_ml=usar_ml)
    resultado = agente.invoke({"messages": [{"role": "user", "content": texto}]})

    mensajes = resultado["messages"]
    log = _formatear_log(mensajes)
    respuesta_agente = mensajes[-1].content if mensajes else ""
    analisis = f"{log}\n\nConclusión preliminar del agente:\n{respuesta_agente}"
    clasificacion = cadena_estructurada.invoke({"texto": texto, "analisis": analisis})

    return {
        "clasificacion": clasificacion.model_dump(),
        "log_razonamiento": log,
        "respuesta_agente": respuesta_agente,
    }


if __name__ == "__main__":
    import json

    tickets_demo = [
        "Buenos dias, no puedo inscribir Machine Learning porque no hay cupos y manana cierra la matricula.",
        "Necesito un certificado de notas urgente para una beca que cierra hoy.",
        "Olvide mi contrasena del SIA y no me llega el correo de recuperacion.",
    ]
    for t in tickets_demo:
        print("\n" + "=" * 80)
        print("TICKET:", t)
        print("=" * 80)
        salida = procesar_ticket(t)
        print("\n--- LOG DE RAZONAMIENTO (CoT) ---")
        print(salida["log_razonamiento"])
        print("\n--- CLASIFICACION ESTRUCTURADA ---")
        print(json.dumps(salida["clasificacion"], indent=2, ensure_ascii=False))
