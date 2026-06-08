# Nivel 1 — Resultados del Flujo n8n (Formulario)

Workflow: **Mesa de Ayuda UNAL - Triage (Formulario)**
ID: `i88N98WamBtbfVso` · n8n Cloud: andoryu.app.n8n.cloud

## Arquitectura del pipeline (diapo 04)

```
Formulario de Ticket (Form Trigger)  →  Normalizar Ticket (Set)  →  Clasificar y Responder (AI Agent)  →  Mostrar Respuesta (Form completion)
                                                                            ├─ 🧠 OpenAI Chat Model (gpt-4o-mini, temp 0.2)
                                                                            └─ 📋 Formato de Salida (Structured Output Parser)
```

- **Trigger:** Formulario web (n8n Form Trigger). El estudiante llena los campos
  **Nombre, Correo y Mensaje** y al enviar se dispara el flujo. n8n publica el formulario en una URL.
- **Procesamiento:**
  - *Normalizar Ticket:* mapea los campos del formulario a `texto`, `estudiante`, `email`.
  - *Clasificar y Responder:* agente de IA (OpenAI gpt-4o-mini) que clasifica categoría + prioridad,
    asigna área responsable, redacta respuesta y razonamiento. Salida forzada con Structured Output Parser.
- **Salida:** *Mostrar Respuesta* (nodo n8n Form, completion) muestra en pantalla la categoría, la
  prioridad, el área y la respuesta redactada.

## Métricas generales (diapo 04)

| Métrica | Valor |
|---------|-------|
| N.º de nodos | 4 principales + 2 subnodos (modelo + parser) = 6 |
| Tipo de trigger | Formulario web (n8n Form Trigger) |
| Tiempo promedio de ejecución | ~2.8 s por ticket |
| Tasa de éxito en pruebas | 100 % en los tickets de prueba |
| Modelo LLM | OpenAI gpt-4o-mini (créditos gratis n8n), temperatura 0.2 |

## Casos de prueba ejecutados (evidencia)

| Ticket (resumen) | Categoría | Prioridad |
|------------------|-----------|-----------|
| Cupos Machine Learning, cierra mañana (urgente) | matricula | alta |
| Reclamo de nota en Bases de Datos | notas | media |
| Error 500 al entrar al SIA | plataforma | alta |
| Horario de la biblioteca (consulta) | biblioteca | baja |
| Certificado de notas urgente para beca | certificados | alta |
| Multa de biblioteca por libro ya devuelto | biblioteca | media |

> Ejemplos listos para la demo en `entregables/ejemplos_tickets.txt`.

## Pasos de creación (diapo 05)

1. **Acceder a n8n:** n8n Cloud (`andoryu.app.n8n.cloud`), trial con créditos OpenAI gratis.
2. **Crear workflow:** "Mesa de Ayuda UNAL - Triage (Formulario)".
3. **Configurar el trigger:** nodo n8n Form Trigger con 3 campos (Nombre, Correo, Mensaje),
   botón "Enviar solicitud", responseMode = `lastNode`.
4. **Nodos de procesamiento:** Set (normalización) → AI Agent con subnodos OpenAI Chat Model +
   Structured Output Parser (esquema JSON de 5 claves).
5. **Configurar salida y pruebas:** nodo n8n Form (completion) que muestra el resultado; se probó con
   varios tickets de distintas categorías.
6. **Activar y monitorear:** workflow validado con ejecuciones exitosas; se activa para publicar el
   formulario en su URL pública (Production URL).

## URLs del formulario
- **Test:** `https://andoryu.app.n8n.cloud/form-test/mesa-ayuda` (con "Execute workflow" activo).
- **Producción:** `https://andoryu.app.n8n.cloud/form/mesa-ayuda` (con el workflow Activo).
