# -*- coding: utf-8 -*-
"""
NIVEL 3 — Evaluación cuantitativa: ¿qué aporta integrar el ML al agente?

Compara el agente LangChain en dos configuraciones sobre una muestra del dataset:
  - SIN ML (línea base Nivel 2): el agente clasifica usando solo el LLM + RAG.
  - CON ML (Nivel 3): el agente dispone además de la herramienta del clasificador ML.

Mide accuracy de categoría y tiempo de respuesta, guarda los resultados en
docs/nivel3_comparacion.json y genera un gráfico de barras para la diapositiva 11.

Uso:
    python evaluar.py            # muestra por defecto (30 tickets)
    python evaluar.py 50         # muestra de 50 tickets
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agente import procesar_ticket

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
_DATA = os.path.join(_RAIZ, "data", "tickets_unal.csv")
_DOCS = os.path.join(_RAIZ, "docs")
_SALIDA_JSON = os.path.join(_DOCS, "nivel3_comparacion.json")
_SALIDA_PNG = os.path.join(_DOCS, "nivel3_comparacion.png")


def evaluar_config(tickets, etiquetas, usar_ml):
    aciertos, tiempos = 0, []
    for texto, real in zip(tickets, etiquetas):
        t0 = time.time()
        try:
            salida = procesar_ticket(texto, usar_ml=usar_ml)
            pred = salida["clasificacion"]["categoria"]
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {e}")
            pred = "error"
        tiempos.append(time.time() - t0)
        if pred == real:
            aciertos += 1
    n = len(tickets)
    return {
        "accuracy": round(aciertos / n, 4),
        "aciertos": aciertos,
        "total": n,
        "tiempo_promedio_s": round(sum(tiempos) / n, 2),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    df = pd.read_csv(_DATA).sample(n=n, random_state=7).reset_index(drop=True)
    tickets = df["texto"].tolist()
    etiquetas = df["categoria"].tolist()

    print(f"Evaluando sobre {n} tickets...\n")
    print(">>> Configuración BASE (Nivel 2, sin ML)")
    base = evaluar_config(tickets, etiquetas, usar_ml=False)
    print(f"    accuracy={base['accuracy']}  tiempo={base['tiempo_promedio_s']}s\n")

    print(">>> Configuración CON ML (Nivel 3)")
    con_ml = evaluar_config(tickets, etiquetas, usar_ml=True)
    print(f"    accuracy={con_ml['accuracy']}  tiempo={con_ml['tiempo_promedio_s']}s\n")

    mejora = round((con_ml["accuracy"] - base["accuracy"]) * 100, 2)
    resultados = {
        "muestra": n,
        "nivel2_base_sin_ml": base,
        "nivel3_con_ml": con_ml,
        "mejora_accuracy_puntos_pct": mejora,
    }
    os.makedirs(_DOCS, exist_ok=True)
    with open(_SALIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados -> {_SALIDA_JSON}")
    print(f"Mejora en accuracy: {mejora:+.2f} puntos porcentuales")

    # --- Gráfico comparativo ---
    fig, ax = plt.subplots(figsize=(7, 5))
    configs = ["Nivel 2\n(LLM + RAG)", "Nivel 3\n(+ ML)"]
    accs = [base["accuracy"], con_ml["accuracy"]]
    barras = ax.bar(configs, accs, color=["#9aa0a6", "#1a73e8"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy de categoría")
    ax.set_title(f"Impacto de integrar el ML al agente (n={n})")
    for b, a in zip(barras, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.0%}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(_SALIDA_PNG, dpi=150)
    print(f"Gráfico guardado -> {_SALIDA_PNG}")


if __name__ == "__main__":
    main()
