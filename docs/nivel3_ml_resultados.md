# Nivel 3 — Técnica de Racionalidad: Machine Learning

## Justificación de la técnica (diapo 10)

Se eligió **Machine Learning (clasificación supervisada de texto)** como técnica de racionalidad
formal porque el problema central del caso de uso —enrutar correctamente cada ticket— es justamente
un problema de clasificación con datos etiquetados, y permite reportar **métricas cuantitativas
estándar** (accuracy, precision, recall, F1, matriz de confusión) que la rúbrica exige. Frente a
Lógica Difusa o Sistema Experto (basados en reglas manuales), el modelo **aprende** los patrones del
lenguaje de los estudiantes y generaliza a redacciones nuevas.

## Diseño del modelo (diapo 10)

| Elemento | Detalle |
|----------|---------|
| Algoritmo | **Regresión Logística** (mejor por F1-macro), comparado contra Random Forest y XGBoost |
| Features | **TF-IDF** (unigramas + bigramas, sublinear_tf, stopwords en español) |
| Tipo | Supervisado, multiclase (5 categorías) |
| Dataset | 935 tickets sintéticos etiquetados (con ambigüedad, typos y ruido de etiqueta para realismo) |
| Validación | Hold-out 80/20 (748 train / 187 test) + validación cruzada 5-fold |
| Pipeline | `TfidfVectorizer → Clasificador` (scikit-learn Pipeline), serializado con joblib |

## Resultados del clasificador (diapo 11)

Comparación de algoritmos (test hold-out, 187 tickets):

| Modelo | Accuracy | F1-macro | CV F1-macro (5-fold) |
|--------|----------|----------|----------------------|
| **Regresión Logística** ✅ | **0.9733** | **0.9731** | 0.9626 |
| Random Forest | 0.9679 | 0.9678 | 0.9556 |
| XGBoost | 0.9572 | 0.9566 | 0.9360 |

- Métricas por clase y matriz de confusión: `ml_clasificador/metricas.json`, `ml_clasificador/matriz_confusion.png`.
- Gráfico comparativo de modelos: `ml_clasificador/comparacion_modelos.png`.

## Integración en el pipeline LangChain (diapo 11)

El modelo entrenado se expone como la **herramienta `clasificar_ticket_ml`** del agente LangChain
(`langchain/herramientas.py`). El agente la invoca como primer paso de su razonamiento (CoT) antes de
consultar el RAG y redactar la respuesta. Así, la decisión de categoría deja de depender solo del
criterio del LLM y se apoya en un modelo entrenado y evaluado formalmente.

## Análisis cuantitativo: ¿qué aporta el ML? (diapo 11)

Se evaluó el agente sobre una muestra de 20 tickets en dos configuraciones:

| Configuración | Accuracy de categoría | Tiempo promedio |
|---------------|----------------------|-----------------|
| **Nivel 2** — agente LLM (qwen2.5:3b) + RAG, **sin** ML | 50 % (10/20) | 9.08 s |
| **Nivel 3** — agente **con** la herramienta ML | **95 % (19/20)** | 10.39 s |
| **Mejora** | **+45 puntos porcentuales** | +1.31 s |

> Gráfico: `docs/nivel3_comparacion.png` · datos: `docs/nivel3_comparacion.json`.

**Conclusión:** integrar el clasificador ML casi **duplica la precisión** del agente (de 50 % a 95 %)
con un costo de tiempo despreciable (+1.3 s). El LLM pequeño por sí solo es un clasificador mediocre;
apoyado en la técnica de racionalidad formal se vuelve confiable. Esta es la mejora cuantificable del
Nivel 3 respecto al Nivel 2.

## Limitaciones y trabajo futuro (diapo 11)

- El dataset es **sintético**; en producción debe reentrenarse con tickets reales etiquetados.
- El 97 % del clasificador se mide sobre datos de la misma distribución; redacciones muy distintas
  podrían bajar el desempeño (deriva de dominio).
- Mejora futura: reentrenamiento periódico con realimentación de los operadores, y umbral de confianza
  para escalar a revisión humana los casos donde el modelo tenga baja certeza.
