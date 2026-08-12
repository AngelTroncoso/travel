"""Panel TMC - Orquestador Multiagente (prototipo LangGraph + Groq).

Pivote de arquitectura para el prototipo: en vez de FastAPI + Redis
Streams, este frontend centraliza toda la orquestacion en un grafo de
LangGraph (StateGraph) corriendo dentro de la propia app de Streamlit.
El motor de razonamiento de cada nodo es ChatGroq (rapido y gratuito en
el tier de Groq), leyendo la API key desde GROQ_API_KEY en el .env.

Flujo del grafo:
    Sandbox (sidebar) -> Agente Transporte -> Agente Alojamiento
                       -> Agente Finanzas -> Agente Agenda (orquestador)
                       -> Salida final

Cada nodo anade su razonamiento a una lista de "pasos" en el estado
compartido, que la interfaz de Streamlit muestra paso a paso.
"""

from __future__ import annotations

import os
import re
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

st.set_page_config(page_title="TMC Elite Orquestador", layout="wide")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_ID = os.environ.get("OPENAI_CHAT_MODEL_ID", "openai/gpt-oss-120b")


# --- Herramientas simuladas -------------------------------------------------


@tool
def calcular_tiempo_traslado(origen: str, destino: str) -> str:
    """Calcula (de forma simulada) el tiempo de traslado en minutos entre
    el aeropuerto/origen y un hotel/destino. Devuelve un valor ficticio,
    pensado para demoear la herramienta sin depender de una API real de
    mapas."""
    return (
        f"Tiempo estimado de traslado entre '{origen}' y '{destino}': "
        "35 minutos (valor simulado, sin API real de mapas)."
    )


@tool
def consultar_disponibilidad_hotel(hotel: str, hora_llegada_estimada: str) -> str:
    """Consulta (de forma simulada) si un hotel tiene disponibilidad para
    una llegada tardia/reprogramada. Devuelve un resultado ficticio,
    pensado para demoear la herramienta sin depender de una API real de
    inventario hotelero."""
    return (
        f"Habitacion confirmada para llegada tardia en '{hotel}' "
        f"(hora estimada {hora_llegada_estimada}). Late check-in garantizado, "
        "sin costo adicional (valor simulado, sin API real de inventario)."
    )


ALOJAMIENTO_TOOLS = [calcular_tiempo_traslado, consultar_disponibilidad_hotel]
ALOJAMIENTO_TOOLS_BY_NAME = {t.name: t for t in ALOJAMIENTO_TOOLS}


# --- Estado compartido del grafo --------------------------------------------


class AgendaState(TypedDict):
    problema_usuario: str
    nivel_pasajero: str
    motivo_viaje: str
    pasos: list[dict[str, str]]
    analisis_transporte: str
    analisis_alojamiento: str
    analisis_finanzas: str
    decision_final: str


def _get_llm() -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL_ID, temperature=0.3)


# --- Nodos / agentes del grafo -----------------------------------------------


def nodo_transporte(state: AgendaState) -> AgendaState:
    """Agente Transporte: experto en vuelos. Interpreta el problema del
    usuario desde la perspectiva del itinerario aereo."""
    llm = _get_llm()
    system = SystemMessage(
        content=(
            "ERES UN EXPERTO EN AVIACIÓN. REGLA CRÍTICA Y ESTRICTA: LA "
            "PALABRA \"SECADORA\" ESTÁ ABSOLUTAMENTE PROHIBIDA EN TU "
            "VOCABULARIO. DEBES USAR ÚNICAMENTE LOS TÉRMINOS \"AEROLÍNEA\", "
            "\"LÍNEA AÉREA\" O \"COMPAÑÍA AÉREA\".\n\n"
            "Eres el Agente de Transporte de una TMC (Travel Management "
            "Company). Eres experto en vuelos, retrasos y reprogramaciones. "
            "Analiza el problema del viajero SOLO desde la perspectiva del "
            "vuelo: que paso, que tan grave es, y que opciones de vuelo "
            "existen. Responde en 2-4 frases, en español, de forma tecnica "
            "y concisa.\n\n"
            "REGLA ESTRICTA DE VOCABULARIO: usa exclusivamente vocabulario "
            "tecnico de aviacion (aerolinea, PNR, reubicacion, pasajero). "
            "Esta estrictamente prohibido usar traducciones erroneas como "
            "'secadora' o cualquier termino fuera de contexto de la "
            "industria aeronautica."
        )
    )
    human = HumanMessage(content=state["problema_usuario"])
    respuesta = llm.invoke([system, human])
    analisis = respuesta.content

    pasos = state.get("pasos", [])
    pasos.append({"agente": "Agente de Transporte", "pensamiento": analisis})

    return {**state, "analisis_transporte": analisis, "pasos": pasos}


def nodo_alojamiento(state: AgendaState) -> AgendaState:
    """Agente Alojamiento: experto en ubicar hoteles, calcular
    distancias/traslados y confirmar disponibilidad, usando las
    herramientas calcular_tiempo_traslado y consultar_disponibilidad_hotel.

    Si el LLM decide usar una o mas herramientas, se ejecutan y su
    resultado se devuelve al LLM (segunda vuelta) para que genere la
    recomendacion final en texto, en vez de quedarse solo con la llamada
    a la herramienta sin explicacion.
    """
    llm = _get_llm().bind_tools(ALOJAMIENTO_TOOLS)

    system = SystemMessage(
        content=(
            "Eres el Agente de Alojamiento de una TMC. Eres experto en "
            "ubicar hoteles cercanos y confirmar su disponibilidad. Tienes "
            "disponibles las herramientas 'calcular_tiempo_traslado' (para "
            "estimar el traslado entre el aeropuerto y un hotel candidato) "
            "y 'consultar_disponibilidad_hotel' (para confirmar si el hotel "
            "tiene habitacion para una llegada tardia/reprogramada). Usa las "
            "herramientas que sean relevantes y luego da tu recomendacion "
            "final de alojamiento en 2-4 frases, en español."
        )
    )
    human = HumanMessage(
        content=(
            f"Problema original del viajero: {state['problema_usuario']}\n"
            f"Analisis del Agente de Transporte: {state['analisis_transporte']}\n\n"
            "Evalua opciones de hotel cercanas al aeropuerto, estima el "
            "traslado y confirma disponibilidad usando las herramientas "
            "disponibles."
        )
    )

    respuesta = llm.invoke([system, human])

    tool_calls = getattr(respuesta, "tool_calls", None) or []
    resultados_tools: list[str] = []

    for call in tool_calls:
        nombre_tool = call.get("name")
        herramienta = ALOJAMIENTO_TOOLS_BY_NAME.get(nombre_tool)
        if herramienta is None:
            continue
        resultado_tool = herramienta.invoke(call.get("args", {}))
        resultados_tools.append(f"[{nombre_tool}] {resultado_tool}")

    if tool_calls:
        # Segunda vuelta con una conversacion NUEVA y sin herramientas
        # vinculadas: se le pasa al LLM el resultado de las herramientas
        # como texto plano dentro de un HumanMessage. Esto evita el error
        # de Groq "tool choice is none, but model called a tool", que
        # ocurre cuando el historial conserva tool_calls previos aunque se
        # use tool_choice="none": el modelo puede igual intentar otro
        # tool_call. Al no tener herramientas bind_tools en esta llamada,
        # es fisicamente imposible que la respuesta contenga un tool_call.
        llm_solo_texto = _get_llm()
        system_resumen = SystemMessage(
            content=(
                "Eres el Agente de Alojamiento de una TMC. Ya consultaste "
                "las herramientas necesarias; ahora debes redactar tu "
                "recomendacion final de alojamiento en 2-4 frases, en "
                "español, basandote unicamente en los resultados que se te "
                "entregan a continuacion. No tienes herramientas "
                "disponibles en este paso, solo debes responder en texto."
            )
        )
        human_resumen = HumanMessage(
            content=(
                f"Problema original del viajero: {state['problema_usuario']}\n"
                f"Analisis del Agente de Transporte: {state['analisis_transporte']}\n\n"
                "Resultados de las consultas realizadas:\n"
                + "\n".join(resultados_tools)
                + "\n\nRedacta la recomendacion final de alojamiento."
            )
        )
        respuesta_final = llm_solo_texto.invoke([system_resumen, human_resumen])
        analisis = (respuesta_final.content or "").strip()
    else:
        analisis = (respuesta.content or "").strip()

    if not analisis:
        analisis = "Sin recomendacion adicional de alojamiento."
    if resultados_tools:
        analisis = f"{analisis}\n\n" + "\n".join(resultados_tools)

    pasos = state.get("pasos", [])
    pasos.append({"agente": "Agente de Alojamiento", "pensamiento": analisis})

    return {**state, "analisis_alojamiento": analisis, "pasos": pasos}


def nodo_finanzas(state: AgendaState) -> AgendaState:
    """Agente Finanzas: auditor financiero de la TMC. Evalua el estado del
    vuelo (retraso) y la tarifa, estimando compensaciones o alertando
    sobre costos adicionales segun las reglas de negocio."""
    llm = _get_llm()
    system = SystemMessage(
        content=(
            "Eres el auditor financiero de la TMC. Recibes el estado del "
            "vuelo y la tarifa. Si el retraso supera los 180 minutos, "
            "calcula una compensacion estimada. Si la tarifa es Restrictiva "
            "y el hotel se cancelo, alerta sobre posibles costos "
            "adicionales.\n\n"
            "REGLA VIP: Si el nivel del pasajero es C-Level/VIP, el "
            "presupuesto es ilimitado; aprueba cualquier costo necesario "
            "para asegurar su llegada.\n\n"
            "Responde en 2-4 frases, en español, de forma tecnica y "
            "concisa."
        )
    )
    human = HumanMessage(
        content=(
            f"Problema/escenario original: {state['problema_usuario']}\n\n"
            f"Nivel del pasajero: {state['nivel_pasajero']}\n"
            f"Motivo del viaje: {state['motivo_viaje']}\n\n"
            f"Analisis del Agente de Transporte: {state['analisis_transporte']}\n\n"
            f"Analisis del Agente de Alojamiento: {state['analisis_alojamiento']}\n\n"
            "Evalua el impacto financiero de este escenario."
        )
    )
    respuesta = llm.invoke([system, human])
    analisis = (respuesta.content or "").strip()
    if not analisis:
        analisis = "Sin hallazgos financieros adicionales."

    pasos = state.get("pasos", [])
    pasos.append({"agente": "Agente de Finanzas", "pensamiento": analisis})

    return {**state, "analisis_finanzas": analisis, "pasos": pasos}


def nodo_agenda(state: AgendaState) -> AgendaState:
    """Agente Agenda (orquestador): evalua el estado general aportado por
    Transporte, Alojamiento y Finanzas, y decide si REPROGRAMAR o
    ESCALAR.

    Actua como un Orquestador Enterprise: su salida es un reporte
    ejecutivo en Markdown estructurado (sin texto conversacional), que
    se muestra al usuario en una seccion final dedicada, no dentro del
    log paso a paso.
    """
    llm = _get_llm()
    system = SystemMessage(
        content=(
            "Eres el Orquestador Enterprise de viajes corporativos "
            "(Agente de Agenda) de una TMC. Recibes el analisis del "
            "Agente de Transporte, del Agente de Alojamiento y del Agente "
            "de Finanzas. Debes decidir si REPROGRAMAR el itinerario "
            "automaticamente o ESCALAR el conflicto a un humano.\n\n"
            "REGLA DE URGENCIA: Si el motivo del viaje es 'Reunión de "
            "Directorio Crítica', la velocidad es mas importante que el "
            "costo. Adapta tu decision (REPROGRAMAR o ESCALAR) basandote "
            "en el perfil del pasajero y la urgencia del viaje.\n\n"
            "FORMATO DE SALIDA OBLIGATORIO: tu respuesta debe estar "
            "formateada EXCLUSIVAMENTE en Markdown estructurado, sin "
            "texto conversacional, saludos ni introducciones. Debe "
            "incluir EXACTAMENTE estas tres secciones, en este orden y "
            "con estos encabezados literales:\n\n"
            "### 🎯 Decisión Final: [REPROGRAMAR o ESCALAR]\n"
            "(Reemplaza el contenido entre corchetes por la palabra "
            "REPROGRAMAR o ESCALAR, seguida de una justificacion breve.)\n\n"
            "### 📋 Plan de Acción Operativo\n"
            "(Vinetas con los pasos concretos a seguir respecto al vuelo "
            "y al hotel.)\n\n"
            "### 💰 Resumen de Impacto Financiero\n"
            "(Basado unicamente en lo que informo el Agente de Finanzas.)\n\n"
            "Responde en español."
        )
    )
    human = HumanMessage(
        content=(
            f"Problema original del viajero: {state['problema_usuario']}\n\n"
            f"Nivel del pasajero: {state['nivel_pasajero']}\n"
            f"Motivo del viaje: {state['motivo_viaje']}\n\n"
            f"Analisis del Agente de Transporte: {state['analisis_transporte']}\n\n"
            f"Analisis del Agente de Alojamiento: {state['analisis_alojamiento']}\n\n"
            f"Analisis del Agente de Finanzas: {state['analisis_finanzas']}\n\n"
            "Genera el reporte ejecutivo final."
        )
    )
    respuesta = llm.invoke([system, human])
    decision = (respuesta.content or "").strip()

    pasos = state.get("pasos", [])
    pasos.append(
        {
            "agente": "Agente de Agenda (Orquestador)",
            "pensamiento": "Evaluando variables y generando reporte ejecutivo final...",
        }
    )

    return {**state, "decision_final": decision, "pasos": pasos}


# --- Construccion del grafo --------------------------------------------------


def construir_grafo():
    grafo = StateGraph(AgendaState)

    grafo.add_node("transporte", nodo_transporte)
    grafo.add_node("alojamiento", nodo_alojamiento)
    grafo.add_node("finanzas", nodo_finanzas)
    grafo.add_node("agenda", nodo_agenda)

    grafo.set_entry_point("transporte")
    grafo.add_edge("transporte", "alojamiento")
    grafo.add_edge("alojamiento", "finanzas")
    grafo.add_edge("finanzas", "agenda")
    grafo.add_edge("agenda", END)

    return grafo.compile()


# --- Diagrama del flujo (Mermaid) --------------------------------------------


MERMAID_DIAGRAM = """
graph LR
    A[Sandbox del Usuario] --> T[Agente Transporte]
    T --> H[Agente Alojamiento]
    H --> F[Agente Finanzas]
    F --> AG[Agente Agenda - Orquestador]
    AG --> S[Salida final]
"""


def render_mermaid(diagram: str, height: int = 220) -> None:
    """Renderiza un diagrama Mermaid embebiendo mermaid.js via CDN.

    Streamlit no tiene soporte nativo para Mermaid, asi que se usa
    st.iframe con un string HTML que carga mermaid.js desde CDN.
    """
    html = f"""
    <div class="mermaid">
    {diagram}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
    </script>
    """
    st.iframe(html, height=height)


# --- Exportacion a PDF --------------------------------------------------------


# Los emojis usados en el reporte no son soportados por las fuentes core
# de PDF (Helvetica/Arial solo cubren latin-1). Se reemplazan por texto
# plano equivalente antes de escribir, y cualquier otro caracter fuera de
# rango se sustituye por "?" para evitar que fpdf2 falle al renderizar.
_EMOJI_REPLACEMENTS = {
    "🎯": "[Decisión]",
    "📋": "[Plan]",
    "💰": "[Financiero]",
    "📑": "[Reporte]",
    "📥": "[Descargar]",
}


def _sanear_texto_pdf(texto: str) -> str:
    for emoji, reemplazo in _EMOJI_REPLACEMENTS.items():
        texto = texto.replace(emoji, reemplazo)
    return texto.encode("latin-1", errors="replace").decode("latin-1")


def generar_pdf(texto_reporte: str) -> bytes:
    """Genera un PDF simple con el texto del reporte ejecutivo final.

    El texto viene en Markdown (encabezados ### y vinetas -), pero fpdf2
    no interpreta Markdown, asi que se escribe como texto plano linea
    por linea usando multi_cell para el ajuste automatico de lineas. Los
    emojis y caracteres fuera de latin-1 se sanean antes de escribir,
    porque las fuentes core de fpdf2 no los soportan.
    """
    pdf = FPDF()
    pdf.add_page()

    # new_x/new_y="LMARGIN"/"NEXT" fuerza a que el cursor vuelva al
    # margen izquierdo y baje una linea despues de cada multi_cell; sin
    # esto, el cursor X queda pegado al borde derecho y la siguiente
    # celda no tiene espacio horizontal para renderizar (fpdf2 lanza
    # "Not enough horizontal space to render a single character").
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.multi_cell(
        0,
        10,
        _sanear_texto_pdf("Reporte Ejecutivo de Resolución - TMC"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", size=12)

    for linea in texto_reporte.split("\n"):
        linea_saneada = _sanear_texto_pdf(linea).strip()
        if linea_saneada:
            pdf.multi_cell(0, 8, linea_saneada, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.ln(4)

    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        return pdf_bytes.encode("latin-1", "replace")
    return bytes(pdf_bytes)


# --- Extraccion de metricas desde el reporte final --------------------------


def extraer_decision(texto: str) -> str:
    """Extrae REPROGRAMAR o ESCALAR (lo que aparezca primero) del reporte,
    para mostrarlo como metrica destacada en la UI."""
    upper = texto.upper()
    candidatos = [
        (upper.find(palabra), palabra)
        for palabra in ("REPROGRAMAR", "ESCALAR")
        if upper.find(palabra) != -1
    ]
    if not candidatos:
        return "N/D"
    candidatos.sort(key=lambda item: item[0])
    return candidatos[0][1]


def extraer_impacto_financiero(texto: str) -> str:
    """Extrae un valor monetario (o etiqueta equivalente) de la seccion
    de Impacto Financiero, para mostrarlo como metrica destacada."""
    match_seccion = re.search(r"Impacto Financiero[^\n]*\n(.*)", texto, re.S)
    seccion = match_seccion.group(1) if match_seccion else texto

    match_monto = re.search(r"\$\s?[\d][\d.,]*\s?(?:USD|EUR)?", seccion)
    if match_monto:
        return match_monto.group(0).strip()

    if re.search(r"sin l[ií]mite|ilimitado", seccion, re.IGNORECASE):
        return "Sin límite (VIP)"

    return "Ver detalle"


# --- Interfaz Streamlit -------------------------------------------------------


st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A2342 0%, #1E3A8A 100%);
    }
    [data-testid="stSidebar"] * {
        color: #F3F4F6;
    }
    /* Inputs de selectbox y slider dentro del sidebar: texto negro nítido
       para máxima legibilidad sobre el fondo blanco de los controles. */
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="select"] svg,
    [data-testid="stSidebar"] [data-baseweb="select"] path,
    [data-testid="stSidebar"] input {
        color: #111111 !important;
        fill: #111111 !important;
    }
    /* Opciones del dropdown (menú desplegable abierto) */
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] * {
        color: #111111 !important;
    }

    div.stButton > button, div.stDownloadButton > button {
        background-color: #1E3A8A;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        transition: background-color 0.2s ease, transform 0.1s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #2563EB;
        color: #FFFFFF;
        transform: translateY(-1px);
    }

    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero banner asimetrico: imagen izquierda (60%) / titulo derecha (40%) --
_col_img, _col_titulo = st.columns([3, 2], gap="large")

with _col_img:
    st.image("frontend/assets/077.jpg", width="stretch")

with _col_titulo:
    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100%;
            padding: 2rem 1rem;
        ">
            <h1 style="
                color: #1E3A8A;
                font-weight: 800;
                font-size: 2.2rem;
                line-height: 1.2;
                margin-bottom: 1rem;
            ">✈️ TMC Elite<br>Gestor Autónomo<br>de Retrasos</h1>
            <p style="
                font-size: 1.05rem;
                color: #6B7280;
                line-height: 1.6;
                margin-bottom: 1.5rem;
            ">Auditoría financiera, reubicación de vuelos y ajuste de hoteles
            en tiempo real mediante Inteligencia Artificial.</p>
            <p style="
                font-size: 0.85rem;
                color: #9CA3AF;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            ">Powered by LangGraph · Groq · Streamlit</p>
            <p style="
                font-size: 0.85rem;
                color: #9CA3AF;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-top: 0.25rem;
            ">Soporta múltiples opciones dinámicas (hasta 5+ perfiles y tarifas diversas)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_mermaid(MERMAID_DIAGRAM)

if not GROQ_API_KEY:
    st.error(
        "No se encontro GROQ_API_KEY en el entorno (.env). "
        "Configura tu API key de Groq para poder ejecutar el grafo."
    )

# --- Sidebar: Sandbox de simulacion de escenarios ---------------------------

st.sidebar.header("Sandbox de Escenario")

minutos_retraso = st.sidebar.slider(
    "Minutos de Retraso", min_value=0, max_value=600, value=180, step=5
)

estado_hotel_original = st.sidebar.selectbox(
    "Estado original del Hotel",
    options=[
        "Confirmado",
        "Cancelado por no-show",
        "Reservación Confirmada (Suite Premium, Caso Complejo)",
        "En Lista de Espera",
        "Modificación Pendiente de Confirmación",
    ],
)

tipo_tarifa_vuelo = st.sidebar.selectbox(
    "Tipo de Tarifa de Vuelo",
    options=[
        "Flexible",
        "Económica Restrictiva",
        "Económica Flexible - Tarifa V (Reembolsable)",
        "Business Full Flex",
        "Tarifa Corporativa Negociada",
    ],
)

nivel_pasajero = st.sidebar.selectbox(
    "Nivel del Pasajero",
    options=[
        "C-Level / VIP",
        "Ejecutivo Estándar",
        "Socio VIP Gold Plus (Tier 2)",
        "Frecuente Platinum",
        "Invitado Corporativo",
    ],
)

motivo_viaje = st.sidebar.selectbox(
    "Motivo del Viaje",
    options=[
        "Reunión de Directorio Crítica",
        "Asistencia a Conferencia",
        "Reunión Crítica de Directorio con Inversores",
        "Visita Comercial Estratégica",
        "Trabajo Interno",
    ],
)

simular = st.sidebar.button("Simular Escenario", type="primary", width="stretch")

if simular:
    if not GROQ_API_KEY:
        st.warning("Configura GROQ_API_KEY antes de ejecutar el grafo.")
    else:
        problema_usuario = (
            f"El vuelo del pasajero sufrio un retraso de {minutos_retraso} minutos. "
            f"El estado original de la reserva de hotel era: {estado_hotel_original}. "
            f"La tarifa del vuelo es de tipo: {tipo_tarifa_vuelo}. "
            f"El nivel del pasajero es: {nivel_pasajero}. "
            f"El motivo del viaje es: {motivo_viaje}."
        )

        st.subheader("Escenario simulado")
        st.info(problema_usuario)

        grafo = construir_grafo()
        estado_inicial: AgendaState = {
            "problema_usuario": problema_usuario,
            "nivel_pasajero": nivel_pasajero,
            "motivo_viaje": motivo_viaje,
            "pasos": [],
            "analisis_transporte": "",
            "analisis_alojamiento": "",
            "analisis_finanzas": "",
            "decision_final": "",
        }

        with st.spinner("Ejecutando el grafo de agentes..."):
            try:
                estado_final = grafo.invoke(estado_inicial)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error ejecutando el grafo: {exc}")
                estado_final = None

        if estado_final is not None:
            st.subheader("Razonamiento paso a paso")
            for i, paso in enumerate(estado_final["pasos"], start=1):
                with st.expander(f"{i}. {paso['agente']}", expanded=True):
                    st.write(paso["pensamiento"])

            st.subheader("📑 Reporte Ejecutivo de Resolución")
            decision = estado_final["decision_final"]

            decision_extraida = extraer_decision(decision)
            impacto_extraido = extraer_impacto_financiero(decision)

            col_metric_1, col_metric_2 = st.columns(2)
            with col_metric_1:
                st.metric(
                    label="Decisión del Orquestador",
                    value=decision_extraida,
                    delta="Automático" if decision_extraida == "REPROGRAMAR" else "Requiere revisión",
                    delta_color="normal" if decision_extraida == "REPROGRAMAR" else "inverse",
                )
            with col_metric_2:
                st.metric(
                    label="Impacto Financiero Estimado",
                    value=impacto_extraido,
                )

            st.markdown(decision)

            st.download_button(
                label="📥 Descargar Reporte en PDF",
                data=generar_pdf(decision),
                file_name="resolucion_tmc.pdf",
                mime="application/pdf",
            )
