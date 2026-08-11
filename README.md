# TMC Elite - Orquestador Multiagente Autónomo

## Descripción

Sistema IA para resolver disrupciones en viajes corporativos para EtMday 2026.

## Arquitectura Técnica

- **LangGraph** — orquestación del grafo de agentes (`StateGraph`).
- **Groq** — motor de inferencia LLM (vía `langchain-groq`), rápido y de costo cero en su tier gratuito.
- **Streamlit** — interfaz interactiva (sandbox de escenarios y visualización del razonamiento).
- **FPDF2** — generación del reporte ejecutivo final en formato PDF descargable.

## Ecosistema de Agentes

El grafo conecta cuatro nodos en secuencia: **Transporte → Alojamiento → Finanzas → Agenda**.

1. **Agente de Transporte** — experto en vuelos, retrasos y reprogramaciones; analiza el problema del viajero desde la perspectiva del itinerario aéreo, usando exclusivamente vocabulario técnico de aviación.
2. **Agente de Alojamiento** — ubica hoteles cercanos, calcula tiempos de traslado y confirma disponibilidad, apoyándose en herramientas simuladas de traslado e inventario hotelero.
3. **Agente de Finanzas** — auditor financiero de la TMC; estima compensaciones cuando el retraso supera los 180 minutos y alerta sobre costos adicionales según el tipo de tarifa y el nivel del pasajero.
4. **Agente de Agenda (Orquestador Enterprise)** — evalúa el estado global aportado por los tres agentes anteriores y decide si **REPROGRAMAR** o **ESCALAR** el conflicto, entregando un reporte ejecutivo estructurado en Markdown.

## Instalación Local

1. Clonar el repositorio:
   ```
   git clone https://github.com/AngelTroncoso/travel.git
   cd travel
   ```

2. Instalar dependencias con `uv`:
   ```
   uv sync
   ```

3. Configurar variables de entorno:
   - Copia `.env.example` a `.env`.
   - Completa `GROQ_API_KEY` con tu API key de Groq (gratuita en [console.groq.com/keys](https://console.groq.com/keys)).
   - Opcionalmente ajusta `OPENAI_CHAT_MODEL_ID` y `OPENAI_BASE_URL` si usas otro proveedor OpenAI-compatible.

4. Ejecutar la app de Streamlit:
   ```
   .venv\Scripts\streamlit.exe run frontend\app.py
   ```
   (En macOS/Linux: `.venv/bin/streamlit run frontend/app.py`)

5. Abrir el navegador en `http://localhost:8501` y usar el sandbox del sidebar para simular escenarios.
