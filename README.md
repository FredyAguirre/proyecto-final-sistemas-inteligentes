# Mesa de Ayuda UNAL — Clasificador y Triage Inteligente de Tickets

Proyecto final del módulo **Sistemas Inteligentes** — Maestría en Administración
de Sistemas, Universidad Nacional de Colombia, Sede Manizales.
Docente: Luis Fernando Castillo Ossa, PhD.

## Caso de uso

Sistema que recibe solicitudes (tickets / PQRS) de estudiantes dirigidas a la
mesa de ayuda universitaria, **clasifica** automáticamente cada solicitud por
**categoría** (`matrícula`, `notas`, `plataforma`, `biblioteca`, `certificados`)
y **prioridad** (`alta`, `media`, `baja`), y **redacta una respuesta** o la
enruta al área correspondiente.

El mismo caso se desarrolla en tres niveles acumulativos:

| Nivel | Tecnología | Qué resuelve |
|-------|-----------|--------------|
| **1 — Base** | **n8n** | Flujo de automatización: un formulario web recibe el ticket → agente LLM clasifica y responde → página de resultado. |
| **2 — Intermedio** | **LangChain** | Reimplementación con un agente (LLM + ChatPromptTemplate + Tools + Agent + CoT + RAG sobre base de conocimiento). |
| **3 — Avanzado** | **Machine Learning** | Clasificador entrenado (TF-IDF + Reg. Logística/RF/XGBoost) integrado como **Tool** del agente, con métricas formales. |

## Estructura del repositorio

```
.
├── data/                        # Dataset
│   ├── generar_dataset.py       # Generador del dataset sintético (semilla fija)
│   └── tickets_unal.csv         # 935 tickets etiquetados (categoría + prioridad)
├── ml_clasificador/             # NIVEL 3 — Machine Learning
│   ├── entrenar.py              # Entrena y compara 3 modelos, guarda métricas y gráficos
│   ├── predecir.py              # Inferencia: clasificar_ticket() — usado como Tool
│   ├── modelo_categoria.joblib  # Modelo entrenado (mejor por F1-macro)
│   ├── metricas.json            # Métricas del clasificador
│   ├── matriz_confusion.png     # Matriz de confusión del mejor modelo
│   └── comparacion_modelos.png  # Comparativo Accuracy / F1 entre modelos
├── langchain/                   # NIVEL 2 — Agente LangChain
│   ├── agente.py                # Agente (LLM + tools + CoT + salida estructurada)
│   ├── herramientas.py          # Tools: clasificador ML + retriever RAG
│   ├── interfaz.py              # Interfaz web (Gradio) del agente
│   ├── evaluar.py               # Comparación Nivel 2 vs Nivel 3
│   └── kb/                       # Base de conocimiento (RAG): 5 documentos UNAL
├── workflow/                    # NIVEL 1 — JSON exportado del flujo n8n
├── docs/                        # Resultados por nivel y gráficos de métricas
├── requirements.txt
└── README.md
```

> El **PPTX** diligenciado, los **videos** y la plantilla original se entregan por
> correo / YouTube-Drive (no se versionan en el repositorio).

## Cómo reproducir

```bash
pip install -r requirements.txt

# 1) Generar el dataset (reproducible, semilla = 42)
python data/generar_dataset.py

# 2) Nivel 3 — entrenar el clasificador ML
python ml_clasificador/entrenar.py
python ml_clasificador/predecir.py    # prueba de inferencia

# 3) Nivel 2/3 — agente LangChain (requiere Ollama local)
#    Instalar Ollama: https://ollama.com/download
ollama pull qwen2.5:3b            # LLM del agente (cabe completo en GPU de 6 GB)
ollama pull nomic-embed-text      # embeddings para el RAG
python langchain/interfaz.py      # interfaz gráfica web (recomendado) -> http://127.0.0.1:7860
python langchain/agente.py        # alternativa por terminal (demo con 3 tickets)
python langchain/evaluar.py 20    # comparación Nivel 2 vs Nivel 3 (genera métricas + gráfico)
```

> **Nivel 1 (n8n):** el flujo vive en n8n Cloud (formulario → agente → respuesta).
> El JSON exportado está en `workflow/`.

## Resultados del clasificador (Nivel 3)

Mejor modelo: **Regresión Logística** (TF-IDF uni+bigramas).

| Métrica (test, hold-out 20%) | Valor |
|------------------------------|-------|
| Accuracy | ~0.97 |
| F1-macro | ~0.97 |
| Precision-macro | ~0.97 |
| Recall-macro | ~0.97 |

Las cifras exactas y el reporte por clase están en `ml_clasificador/metricas.json`.

## Declaración de uso de IA generativa

Conforme a la política de integridad académica del curso, se declara que este
proyecto fue **asistido por IA generativa (Claude)** en:

- Generación del dataset sintético de tickets y del código de entrenamiento.
- Diseño del flujo n8n y del agente LangChain.
- Redacción de documentación.

Todo el código fue **revisado, comprendido y adaptado** al caso de uso real por
el equipo, que puede explicar en detalle cualquier fragmento durante la
sustentación.
