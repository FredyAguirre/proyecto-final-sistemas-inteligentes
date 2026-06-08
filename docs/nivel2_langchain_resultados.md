# Nivel 2 — Agente LangChain (y base del Nivel 3)

## Arquitectura y componentes (diapo 07)

| Componente | Implementación |
|------------|----------------|
| **LLM** | `ChatOllama` — modelo local `qwen2.5:3b` (temp 0.2). Sin costo, sin API key, corre 100% en GPU local (RTX 2060). |
| **ChatPromptTemplate** | System + Human para el paso de salida estructurada (`prompt_estructura`). |
| **Tools** | `clasificar_ticket_ml` (modelo ML, Nivel 3) y `buscar_base_conocimiento` (RAG). |
| **Agent** | `create_agent` (LangChain 1.x sobre langgraph): loop de razonamiento + uso de herramientas. |
| **CoT** | Razonamiento paso a paso estilo ReAct: el agente decide qué herramienta usar y cuándo. Queda en el log. |
| **RAG** | `InMemoryVectorStore` + embeddings `nomic-embed-text` (Ollama) sobre 5 documentos de la base de conocimiento UNAL (15 fragmentos indexados). Retrieval top-k=2. |
| **Salida estructurada** | `llm.with_structured_output(Clasificacion)` con esquema Pydantic (5 campos validados). |

## Decisiones de diseño justificadas (sección 07 / Justificación)

- **¿Por qué Ollama local y no OpenAI en el Nivel 2?** Los créditos gratis de OpenAI del trial de n8n solo
  funcionan dentro de n8n; fuera de allí se requeriría una API key de pago. Ollama local da costo cero,
  reproducibilidad y privacidad (los datos del estudiante no salen del equipo).
- **¿Por qué `qwen2.5:3b` y no 7b?** El 7b (5.1 GB) no cabe completo en los 6 GB de la RTX 2060 y se reparte
  CPU/GPU (~290 s/ticket). El 3b cabe entero en GPU → ~28 s/ticket (~10x más rápido), manteniendo buena
  calidad en español y soporte de tool-calling.
- **¿Por qué RAG?** Para fundamentar las respuestas en procedimientos y contactos oficiales de la UNAL y
  evitar alucinaciones. Se evidenció: el agente cita datos textuales de la base de conocimiento.
- **CoT (ReAct):** el agente razona y encadena herramientas (primero ML, luego RAG) antes de responder.

## Ejemplo de ejecución real — log de razonamiento (diapo 08-09)

Ticket: *"No puedo inscribir Machine Learning porque no hay cupos y mañana cierra la matrícula."*

```
Paso 1: el agente decide usar 'clasificar_ticket_ml' con {'texto': '...'}
   Observación: Categoria predicha por el modelo ML: matricula (confianza 81%).
                Top-3: matricula (81%), certificados (7%), notas (5%).
Paso 2: el agente decide usar 'buscar_base_conocimiento' con {'consulta': 'procedimientos de inscripción a cursos'}
   Observación: [matricula] ... Si una asignatura no muestra cupos, el estudiante puede solicitar
                cupo adicional escribiendo a la Secretaría Académica ...
```

Salida estructurada final:

```json
{
  "categoria": "matricula",
  "prioridad": "alta",
  "area_responsable": "Secretaría Académica",
  "respuesta_sugerida": "Puede escribir directamente a la Secretaría Académica de su facultad indicando el código de la asignatura (Machine Learning) y el grupo al que desea inscribirse. Es importante hacerlo antes de que cierre el plazo para obtener un cupo.",
  "razonamiento": "El modelo ML predice 'matricula' (81%) y la base de conocimiento confirma el procedimiento de cupo adicional. No hay cupos y el plazo cierra mañana, por lo que la prioridad es alta."
}
```

> Nota: la `respuesta_sugerida` cita el procedimiento real recuperado por RAG (escribir a Secretaría
> Académica con el código de la asignatura), demostrando que la respuesta está fundamentada y no alucinada.

## Cadena de pensamiento (CoT) — diapo 08

- **Estrategia:** ReAct (Reasoning + Acting). El agente alterna razonamiento y acciones (llamadas a tools).
- Primero consulta el modelo ML (evidencia cuantitativa), luego el RAG (evidencia documental), y finalmente
  sintetiza la decisión. Esto se evidencia en `intermediate_steps` / mensajes del agente.

## Persistencia y RAG — diapo 08

- **Vector store:** `InMemoryVectorStore` de langchain-core (no requiere FAISS/Chroma externos).
- **Embeddings:** `nomic-embed-text` vía Ollama.
- **Documentos indexados:** 5 archivos Markdown (`langchain/kb/`) → 15 fragmentos (chunk 500, overlap 80).
- **Estrategia de retrieval:** similitud por embeddings, top-k = 2.
