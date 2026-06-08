# -*- coding: utf-8 -*-
"""
Herramientas (Tools) del agente LangChain de la Mesa de Ayuda UNAL.

Define dos herramientas que el agente puede invocar:

1. clasificar_ticket_ml  -> NIVEL 3: consulta el clasificador de Machine Learning
   entrenado (TF-IDF + Regresión Logística) para obtener la categoría del ticket
   con su nivel de confianza. Integra la técnica de racionalidad formal al pipeline.

2. buscar_base_conocimiento -> NIVEL 2 (RAG): recupera los fragmentos más relevantes
   de la base de conocimiento institucional (procedimientos, contactos, plazos) para
   fundamentar la respuesta del agente y reducir alucinaciones.
"""
import glob
import os
import sys

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
_DIR_KB = os.path.join(_AQUI, "kb")

# Permite importar el clasificador ML entrenado en ../ml_clasificador.
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)
from ml_clasificador.predecir import clasificar_ticket as _clasificar_ml  # noqa: E402

# Modelo de embeddings local (Ollama). Debe haberse descargado con:
#   ollama pull nomic-embed-text
MODELO_EMBEDDINGS = "nomic-embed-text"

_retriever = None  # se construye una sola vez (lazy)


def _construir_retriever(k: int = 2):
    """Carga la base de conocimiento, la trocea e indexa en un vector store en memoria."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    documentos = []
    for ruta in sorted(glob.glob(os.path.join(_DIR_KB, "*.md"))):
        categoria = os.path.splitext(os.path.basename(ruta))[0]
        with open(ruta, encoding="utf-8") as f:
            contenido = f.read()
        for fragmento in splitter.split_text(contenido):
            documentos.append(Document(page_content=fragmento,
                                       metadata={"categoria": categoria}))
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)
    vector_store = InMemoryVectorStore.from_documents(documentos, embeddings)
    print(f"[RAG] Base de conocimiento indexada: {len(documentos)} fragmentos.")
    return vector_store.as_retriever(search_kwargs={"k": k})


def _obtener_retriever():
    global _retriever
    if _retriever is None:
        _retriever = _construir_retriever()
    return _retriever


@tool
def clasificar_ticket_ml(texto: str) -> str:
    """Clasifica la categoria de un ticket usando el modelo de Machine Learning entrenado.

    Usa esta herramienta SIEMPRE primero para obtener la categoria probable del ticket.
    Devuelve la categoria predicha y la confianza del modelo (0 a 1).
    """
    r = _clasificar_ml(texto)
    top3 = ", ".join(f"{c} ({p:.0%})" for c, p in r["top3"])
    return (f"Categoria predicha por el modelo ML: {r['categoria']} "
            f"(confianza {r['confianza']:.0%}). Top-3: {top3}.")


@tool
def buscar_base_conocimiento(consulta: str) -> str:
    """Busca informacion oficial en la base de conocimiento de la Mesa de Ayuda UNAL.

    Usa esta herramienta para recuperar procedimientos, plazos y contactos reales antes
    de redactar la respuesta al estudiante. Recibe una consulta en lenguaje natural.
    """
    docs = _obtener_retriever().invoke(consulta)
    if not docs:
        return "No se encontro informacion relevante en la base de conocimiento."
    partes = []
    for d in docs:
        partes.append(f"[{d.metadata.get('categoria', 'general')}] {d.page_content.strip()}")
    return "\n\n---\n\n".join(partes)


HERRAMIENTAS = [clasificar_ticket_ml, buscar_base_conocimiento]
