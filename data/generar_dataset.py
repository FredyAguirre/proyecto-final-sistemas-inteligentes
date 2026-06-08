# -*- coding: utf-8 -*-
"""
Generador de dataset sintético de tickets de la Mesa de Ayuda UNAL.

Crea tickets realistas en español etiquetados con:
  - categoria: matricula | notas | plataforma | biblioteca | certificados
  - prioridad: alta | media | baja

El dataset alimenta tanto al agente (Nivel 1/2) como al clasificador ML (Nivel 3).
Se fija una semilla para que el dataset sea reproducible.
"""
import csv
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Plantillas por categoría. Cada entrada es (texto, prioridad_base).
# La prioridad final se ajusta con marcadores de urgencia más abajo.
# ---------------------------------------------------------------------------
PLANTILLAS = {
    "matricula": [
        ("No puedo inscribir la asignatura {asig} porque el sistema dice que no hay cupos.", "alta"),
        ("Necesito adicionar la materia {asig} pero ya cerró el periodo de adiciones.", "alta"),
        ("Quiero cancelar la asignatura {asig} antes de la fecha límite.", "media"),
        ("¿Cuándo abre el periodo de matrícula para el semestre {sem}?", "baja"),
        ("Me aparece un cruce de horarios entre {asig} y otra materia al matricular.", "media"),
        ("Pagué el recibo de matrícula pero el sistema sigue mostrándome como no matriculado.", "alta"),
        ("Solicito información sobre el proceso de matrícula de la maestría.", "baja"),
        ("No me deja matricular porque tengo una deuda pendiente que ya pagué.", "alta"),
        ("¿Puedo hacer una adición extemporánea de {asig}?", "media"),
        ("Quiero saber cuántos créditos puedo inscribir este semestre.", "baja"),
    ],
    "notas": [
        ("El profesor de {asig} no ha subido las notas y mañana cierra el semestre.", "alta"),
        ("Quiero hacer un reclamo de calificación en la asignatura {asig}.", "media"),
        ("Mi promedio (PAPA) aparece mal calculado en el sistema.", "media"),
        ("¿Dónde puedo consultar mi historia académica con las notas?", "baja"),
        ("La nota que registró el docente de {asig} no coincide con la del examen.", "media"),
        ("Necesito que revisen mi nota de {asig} porque afecta mi grado.", "alta"),
        ("¿Cuándo se publican las notas definitivas del semestre {sem}?", "baja"),
        ("Aparece una asignatura como reprobada pero yo la aprobé.", "alta"),
        ("Solicito el reporte de notas parciales de mis materias.", "baja"),
        ("El sistema no muestra la nota de {asig} aunque ya presenté el examen.", "media"),
    ],
    "plataforma": [
        ("No puedo iniciar sesión en el SIA, me dice usuario o contraseña incorrectos.", "alta"),
        ("Olvidé mi contraseña del correo institucional y no puedo recuperarla.", "media"),
        ("La plataforma SIA se cae cada vez que intento matricular.", "alta"),
        ("¿Cómo activo mi cuenta institucional de la universidad?", "baja"),
        ("El portal académico me muestra un error 500 al ingresar.", "alta"),
        ("No me llega el correo de verificación para restablecer la contraseña.", "media"),
        ("La página de la universidad está muy lenta hoy.", "baja"),
        ("Mi usuario quedó bloqueado tras varios intentos de inicio de sesión.", "alta"),
        ("No puedo acceder a la plataforma Moodle del curso {asig}.", "media"),
        ("¿Cuál es el enlace correcto para entrar al sistema académico?", "baja"),
    ],
    "biblioteca": [
        ("Necesito renovar el préstamo de un libro que vence hoy.", "alta"),
        ("Me aparece una multa en la biblioteca que no reconozco.", "media"),
        ("¿Cómo accedo a las bases de datos científicas desde casa?", "baja"),
        ("Quiero reservar una sala de estudio en la biblioteca.", "baja"),
        ("Devolví un libro pero todavía figura como prestado a mi nombre.", "media"),
        ("No puedo descargar artículos de la base de datos institucional.", "media"),
        ("Tengo un paz y salvo pendiente con la biblioteca y necesito graduarme.", "alta"),
        ("¿Cuál es el horario de atención de la biblioteca?", "baja"),
        ("Solicito ampliar el plazo de préstamo de un libro para mi tesis.", "media"),
        ("El sistema de la biblioteca no me deja hacer el préstamo de un ejemplar.", "media"),
    ],
    "certificados": [
        ("Necesito un certificado de notas con urgencia para una beca que cierra hoy.", "alta"),
        ("Solicito una constancia de estudios para un trámite laboral.", "media"),
        ("¿Cómo descargo el certificado de notas desde el SIA?", "baja"),
        ("Requiero un paz y salvo financiero para iniciar mi grado.", "alta"),
        ("Pedí un certificado hace dos semanas y aún no me lo entregan.", "media"),
        ("¿Cuánto cuesta y cuánto tarda un certificado de estudios?", "baja"),
        ("El certificado que descargué tiene mis datos personales equivocados.", "media"),
        ("Necesito constancia de matrícula para el sisbén / EPS.", "media"),
        ("Solicito el certificado de calificaciones para homologar en otra universidad.", "media"),
        ("¿Dónde solicito el diploma y acta de grado?", "baja"),
    ],
}

ASIGNATURAS = [
    "Cálculo Diferencial", "Programación", "Bases de Datos", "Machine Learning",
    "Redes Neuronales", "Estadística", "Inteligencia Artificial", "Álgebra Lineal",
    "Sistemas Inteligentes", "Minería de Datos", "Procesamiento de Lenguaje Natural",
]
SEMESTRES = ["2026-1", "2026-2", "2027-1"]

# Tickets ambiguos: combinan señales de dos categorías. Se etiquetan con la
# categoría dominante (intención real), pero comparten vocabulario con otra,
# lo que genera confusiones realistas en el clasificador.
AMBIGUOS = [
    ("No puedo entrar al SIA para inscribir {asig}, ¿es problema de la plataforma o de matrícula?", "matricula", "media"),
    ("El sistema no me deja ver mis notas ni descargar el certificado.", "certificados", "media"),
    ("Necesito el certificado pero la plataforma no carga.", "certificados", "alta"),
    ("Quiero reclamar una nota pero el SIA no me deja iniciar sesión.", "notas", "media"),
    ("Voy a graduarme y me falta el paz y salvo de biblioteca y el certificado de notas.", "certificados", "alta"),
    ("La biblioteca dice que tengo multa y por eso no me deja matricular.", "matricula", "alta"),
    ("Tengo problemas para acceder a Moodle y ver mis calificaciones.", "plataforma", "media"),
    ("No aparece mi matrícula y tampoco puedo generar la constancia de estudios.", "matricula", "media"),
    ("Solicito ayuda con un trámite, no sé bien a qué área corresponde.", "plataforma", "baja"),
    ("Buenas, tengo una duda general sobre un proceso de la universidad.", "plataforma", "baja"),
]


def meter_typos(texto, prob=0.25):
    """Introduce errores de tipeo realistas en una fracción de los tickets."""
    if random.random() > prob:
        return texto
    chars = list(texto)
    n = max(1, len(chars) // 60)
    for _ in range(n):
        i = random.randint(0, len(chars) - 1)
        op = random.random()
        if op < 0.4 and chars[i].isalpha():        # duplicar letra
            chars[i] = chars[i] * 2
        elif op < 0.7 and i < len(chars) - 1:        # intercambiar
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif chars[i].isalpha():                      # quitar tilde
            chars[i] = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}.get(chars[i], chars[i])
    return "".join(chars)

# Marcadores que elevan o bajan la prioridad respecto a la base.
URGENTES = ["urgente", "es para hoy", "cierra mañana", "lo necesito ya",
            "tengo plazo hasta hoy", "es muy importante"]
CORTESIA = ["Buenos días,", "Cordial saludo,", "Hola,", "Buenas tardes,",
            "Estimados,", "Apreciado equipo de soporte,"]


def ajustar_prioridad(base):
    """Pequeña variación aleatoria de la prioridad base (ruido realista)."""
    escala = ["baja", "media", "alta"]
    i = escala.index(base)
    r = random.random()
    if r < 0.15 and i > 0:
        return escala[i - 1]
    if r > 0.88 and i < 2:
        return escala[i + 1]
    return base


def _formatear(texto):
    return texto.format(
        asig=random.choice(ASIGNATURAS),
        sem=random.choice(SEMESTRES),
    )


def _ensamblar(texto, prioridad, tid):
    # Inyecta urgencia explícita en algunos casos -> sube a alta.
    if random.random() < 0.18:
        texto = texto + " " + random.choice(URGENTES).capitalize() + "."
        prioridad = "alta"
    # Saludo de cortesía en la mayoría de los tickets.
    if random.random() < 0.7:
        texto = random.choice(CORTESIA) + " " + texto
    texto = meter_typos(texto)
    return {"id": tid, "texto": texto, "prioridad": prioridad}


def generar(n_por_categoria=170):
    filas = []
    tid = 1000
    for categoria, plantillas in PLANTILLAS.items():
        for _ in range(n_por_categoria):
            texto, base = random.choice(plantillas)
            fila = _ensamblar(_formatear(texto), ajustar_prioridad(base), tid)
            fila["categoria"] = categoria
            filas.append(fila)
            tid += 1

    # Tickets ambiguos (~10% del total) que generan confusión realista.
    n_ambig = int(0.10 * len(filas))
    for _ in range(n_ambig):
        texto, categoria, base = random.choice(AMBIGUOS)
        fila = _ensamblar(_formatear(texto), base, tid)
        fila["categoria"] = categoria
        filas.append(fila)
        tid += 1

    # Ruido de etiqueta (~3%): casos mal clasificados por el operador humano,
    # realista en cualquier mesa de ayuda y evita métricas perfectas irreales.
    cats = list(PLANTILLAS.keys())
    for fila in filas:
        if random.random() < 0.03:
            fila["categoria"] = random.choice([c for c in cats if c != fila["categoria"]])

    random.shuffle(filas)
    return filas


def main():
    filas = generar()
    ruta = "tickets_unal.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "texto", "categoria", "prioridad"])
        w.writeheader()
        w.writerows(filas)
    # Resumen por consola.
    from collections import Counter
    cat = Counter(r["categoria"] for r in filas)
    pri = Counter(r["prioridad"] for r in filas)
    print(f"Generados {len(filas)} tickets -> {ruta}")
    print("Por categoria:", dict(cat))
    print("Por prioridad:", dict(pri))


if __name__ == "__main__":
    main()
