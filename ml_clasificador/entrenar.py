# -*- coding: utf-8 -*-
"""
Nivel 3 — Clasificador de Machine Learning para la Mesa de Ayuda UNAL.

Entrena modelos de clasificación de texto que predicen la CATEGORÍA de un
ticket (matricula, notas, plataforma, biblioteca, certificados). Este modelo
se integra después como una Tool del agente LangChain (Nivel 2), de modo que
la decisión de enrutamiento deja de depender solo del LLM y pasa a apoyarse
en un clasificador entrenado y evaluado con métricas formales.

Pipeline:
  1. Carga del dataset etiquetado.
  2. Vectorización TF-IDF (uni+bigramas, stopwords en español).
  3. Comparación de 3 algoritmos: Regresión Logística, Random Forest, XGBoost.
  4. Validación cruzada k-fold + evaluación en hold-out.
  5. Selección del mejor por F1-macro, guardado del modelo y de las métricas.
  6. Gráfico de matriz de confusión para la diapositiva 11.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # backend sin ventana, para guardar PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, "..", "data", "tickets_unal.csv")
SALIDA_MODELO = os.path.join(AQUI, "modelo_categoria.joblib")
SALIDA_METRICAS = os.path.join(AQUI, "metricas.json")
SALIDA_MATRIZ = os.path.join(AQUI, "matriz_confusion.png")
SALIDA_COMPARA = os.path.join(AQUI, "comparacion_modelos.png")

# Stopwords en español frecuentes (sklearn no las trae integradas).
STOPWORDS_ES = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero",
    "sus", "le", "ya", "o", "este", "si", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "uno", "ni", "contra", "ese", "eso",
    "ante", "ellos", "e", "esto", "mi", "antes", "algunos", "yo", "otro",
    "es", "me", "te", "tu", "the", "buenos", "dias", "tardes", "hola",
    "cordial", "saludo", "estimados", "apreciado", "equipo", "soporte",
]


def cargar_datos():
    df = pd.read_csv(DATA)
    print(f"Dataset: {len(df)} tickets | categorias: {sorted(df['categoria'].unique())}")
    return df


def construir_pipeline(clasificador):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words=STOPWORDS_ES,
        )),
        ("clf", clasificador),
    ])


def main():
    df = cargar_datos()
    X = df["texto"].values
    y_text = df["categoria"].values

    # XGBoost requiere etiquetas numéricas.
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    clases = list(le.classes_)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidatos = {
        "Regresion Logistica": LogisticRegression(max_iter=1000, C=5.0),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.2,
            subsample=0.9, eval_metric="mlogloss", random_state=42,
        ),
    }

    resultados = {}
    mejor_nombre, mejor_pipe, mejor_f1 = None, None, -1.0

    for nombre, clf in candidatos.items():
        pipe = construir_pipeline(clf)
        # Validación cruzada 5-fold sobre el set de entrenamiento.
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1_macro")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        m = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision_macro": round(precision_score(y_te, pred, average="macro"), 4),
            "recall_macro": round(recall_score(y_te, pred, average="macro"), 4),
            "f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
            "cv_f1_macro_mean": round(float(cv.mean()), 4),
            "cv_f1_macro_std": round(float(cv.std()), 4),
        }
        resultados[nombre] = m
        print(f"\n=== {nombre} ===")
        print(f"  CV F1-macro: {m['cv_f1_macro_mean']} (+/- {m['cv_f1_macro_std']})")
        print(f"  Test  -> acc={m['accuracy']} prec={m['precision_macro']} "
              f"rec={m['recall_macro']} f1={m['f1_macro']}")

        if m["f1_macro"] > mejor_f1:
            mejor_f1 = m["f1_macro"]
            mejor_nombre = nombre
            mejor_pipe = pipe

    print(f"\n>>> Mejor modelo: {mejor_nombre} (F1-macro={mejor_f1})")

    # Reporte detallado del mejor modelo.
    pred_mejor = mejor_pipe.predict(X_te)
    reporte = classification_report(
        y_te, pred_mejor, target_names=clases, output_dict=True
    )
    print("\nReporte por clase (mejor modelo):")
    print(classification_report(y_te, pred_mejor, target_names=clases))

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_te, pred_mejor)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Matriz de Confusión — {mejor_nombre}")
    plt.tight_layout()
    plt.savefig(SALIDA_MATRIZ, dpi=150)
    print(f"\nMatriz de confusión guardada -> {SALIDA_MATRIZ}")

    # --- Gráfico comparativo de modelos ---
    nombres = list(resultados.keys())
    f1s = [resultados[n]["f1_macro"] for n in nombres]
    accs = [resultados[n]["accuracy"] for n in nombres]
    x = np.arange(len(nombres))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, accs, w, label="Accuracy")
    ax.bar(x + w / 2, f1s, w, label="F1-macro")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparación de Modelos — Clasificación de Categoría")
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(SALIDA_COMPARA, dpi=150)
    print(f"Comparación de modelos guardada -> {SALIDA_COMPARA}")

    # --- Guardar modelo + metadatos ---
    joblib.dump({"pipeline": mejor_pipe, "clases": clases}, SALIDA_MODELO)
    print(f"Modelo guardado -> {SALIDA_MODELO}")

    with open(SALIDA_METRICAS, "w", encoding="utf-8") as f:
        json.dump({
            "mejor_modelo": mejor_nombre,
            "clases": clases,
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "comparacion": resultados,
            "reporte_por_clase": reporte,
        }, f, indent=2, ensure_ascii=False)
    print(f"Métricas guardadas -> {SALIDA_METRICAS}")


if __name__ == "__main__":
    main()
