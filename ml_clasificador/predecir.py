# -*- coding: utf-8 -*-
"""
Inferencia del clasificador ML entrenado (Nivel 3).

Expone `clasificar_ticket(texto)` que devuelve la categoría predicha y la
probabilidad (confianza). Esta función se registra como Tool del agente
LangChain en el Nivel 2/3, de modo que el enrutamiento del ticket se apoya
en un modelo entrenado y no solo en el criterio del LLM.
"""
import os

import joblib

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RUTA_MODELO = os.path.join(_AQUI, "modelo_categoria.joblib")
_modelo = None


def _cargar():
    global _modelo
    if _modelo is None:
        if not os.path.exists(_RUTA_MODELO):
            raise FileNotFoundError(
                "Modelo no encontrado. Ejecute primero: python entrenar.py"
            )
        _modelo = joblib.load(_RUTA_MODELO)
    return _modelo


def clasificar_ticket(texto: str) -> dict:
    """Predice la categoría de un ticket y la confianza asociada.

    Returns:
        dict con: categoria, confianza (0-1) y top3 (categoría, prob).
    """
    modelo = _cargar()
    pipe = modelo["pipeline"]
    clases = modelo["clases"]

    probas = pipe.predict_proba([texto])[0]
    orden = probas.argsort()[::-1]
    top3 = [(clases[i], round(float(probas[i]), 4)) for i in orden[:3]]
    return {
        "categoria": clases[orden[0]],
        "confianza": round(float(probas[orden[0]]), 4),
        "top3": top3,
    }


if __name__ == "__main__":
    pruebas = [
        "No puedo iniciar sesión en el SIA para matricular Machine Learning",
        "Necesito un certificado de notas urgente para una beca",
        "El profesor no ha subido las calificaciones del semestre",
        "Quiero renovar el préstamo de un libro de la biblioteca",
    ]
    for t in pruebas:
        r = clasificar_ticket(t)
        print(f"\nTicket: {t}")
        print(f"  -> {r['categoria']} (confianza {r['confianza']:.2%})")
        print(f"     top3: {r['top3']}")
