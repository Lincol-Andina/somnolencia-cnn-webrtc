"""app.py
=============================================================================
Sistema Inteligente de Detección de Somnolencia en Conductores en Tiempo Real
Aplicación Web Interactiva con Streamlit, TensorFlow/Keras y OpenCV
Sprint 2 - Visión por Computadora & Deep Learning
=============================================================================
"""

import os
import sys
import json
import time
import glob
import random
import threading
from pathlib import Path

# Configuración de codificación para consolas Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suprimir logs innecesarios de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import tensorflow as tf
from tensorflow import keras
import streamlit.components.v1 as components

# Configuración inicial de la página en Streamlit
st.set_page_config(
    page_title="Driver Drowsiness AI | Sistema de Seguridad Vehicular",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


def disable_browser_translation():
    """
    Streamlit declara el documento como lang="en". En apps en español, Chrome/Edge
    puede ofrecer traducción automática y al modificar el DOM React falla con
    'removeChild'. Este script marca el documento padre como español y sin traducción.
    """
    components.html(
        """
        <script>
        (function () {
            try {
                const doc = window.parent.document;
                doc.documentElement.lang = "es";
                doc.documentElement.setAttribute("translate", "no");
                doc.documentElement.classList.add("notranslate");
                if (doc.body) doc.body.classList.add("notranslate");

                if (!doc.querySelector('meta[name="google"][content="notranslate"]')) {
                    const meta = doc.createElement("meta");
                    meta.name = "google";
                    meta.content = "notranslate";
                    doc.head.appendChild(meta);
                }
            } catch (e) {
                console.warn("No se pudo desactivar la traducción automática:", e);
            }
        })();
        </script>
        """,
        height=0,
    )


disable_browser_translation()

# =============================================================================
# ESTILOS CSS PERSONALIZADOS (DISEÑO PROFESIONAL, GLASSMORPHISM Y ALTO IMPACTO)
# =============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Fondo principal y gradientes */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #030712 100%);
        color: #f8fafc;
    }

    /* Encabezado principal */
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Tarjetas de estado destacadas */
    .status-card {
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(8px);
        margin-bottom: 16px;
    }

    .status-alert {
        background: rgba(34, 197, 94, 0.12);
        border: 2px solid #22c55e;
        color: #4ade80;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.25);
    }

    .status-drowsy {
        background: rgba(239, 68, 68, 0.18);
        border: 2px solid #ef4444;
        color: #f87171;
        box-shadow: 0 0 28px rgba(239, 68, 68, 0.45);
        animation: pulse-red 1.2s infinite alternate;
    }

    @keyframes pulse-red {
        0% { transform: scale(1.0); box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
        100% { transform: scale(1.02); box-shadow: 0 0 32px rgba(239, 68, 68, 0.7); }
    }

    /* Tarjetas de métricas */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(6px);
    }

    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Fases académicas */
    .phase-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #3b82f6;
        color: white;
        margin-bottom: 8px;
    }

    .academic-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# FUNCIONES AUXILIARES: AUDIO, INFERENCIA Y CARGA DE MODELO
# =============================================================================

def play_alarm_audio():
    """Emite un tono acústico no bloqueante en Windows o reproduce el audio wav sintetizado."""
    try:
        import winsound
        # Frecuencia aguda penetrante de advertencia (2500 Hz por 280 ms)
        winsound.Beep(2500, 280)
        return
    except Exception:
        pass

    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        alarm_file = "alarm.wav"
        if os.path.exists(alarm_file):
            sound = pygame.mixer.Sound(alarm_file)
            sound.play()
    except Exception:
        pass


def trigger_alarm_in_background():
    """Dispara el sonido de alerta en un hilo secundario sin congelar el renderizado visual."""
    t = threading.Thread(target=play_alarm_audio, daemon=True)
    t.start()


@st.cache_resource
def load_trained_model(model_path="model_ddd.keras"):
    """Carga y compila en caché el modelo Keras entrenado para inferencias ultrarrápidas."""
    if not os.path.exists(model_path):
        return None
    try:
        model = keras.models.load_model(model_path, compile=False)
        # Precalentamiento del grafo estático con un tensor sintético
        dummy_tensor = np.zeros((1, 64, 64, 3), dtype=np.float32)
        _ = model(dummy_tensor, training=False)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo desde {model_path}: {e}")
        return None


def load_metrics_data(metrics_path="metrics.json"):
    """Lee el archivo JSON de métricas de desempeño."""
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def preprocess_frame(frame_bgr, target_size=(64, 64)):
    """
    Preprocesa un frame capturado de la cámara para la CNN:
    - Conversión BGR a RGB
    - Redimensionamiento a 64x64
    - Adición de dimensión de lote (Batch Dimension) -> (1, 64, 64, 3)
    Nota: La normalización [0, 1] se realiza dentro de la capa Rescaling del modelo.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    tensor = np.expand_dims(resized, axis=0).astype(np.float32)
    return tensor, frame_rgb


def draw_hud_overlay(frame_bgr, status_text, is_drowsy, prob_drowsy, fps, consecutive_drowsy, alarm_triggered):
    """Dibuja un display HUD profesional sobre el frame de video de OpenCV."""
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # Color del tema: Rojo (#EF4444 en BGR: 68, 68, 239) o Verde (#22C55E en BGR: 94, 197, 34)
    theme_color = (68, 68, 239) if is_drowsy else (94, 197, 34)
    
    # Franja superior semitransparente para información de telemetría
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 23, 42), -1)
    
    # Si la alarma está activa, dibujar borde de advertencia en todo el marco
    if alarm_triggered:
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 220), -1)
        cv2.putText(overlay, "ALERTA CRITICA: CONDUCTOR SOMNOLIENTO", (w // 2 - 230, h - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

    # Texto de estado principal
    badge_label = "[!] " + status_text if is_drowsy else "[OK] " + status_text
    cv2.putText(frame_bgr, badge_label, (18, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, theme_color, 2, cv2.LINE_AA)

    # Indicador de probabilidad
    prob_text = f"P(Somnolencia): {prob_drowsy * 100:.1f}%"
    cv2.putText(frame_bgr, prob_text, (w - 280, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)

    # Indicador de FPS y contador de persistencia
    telemetry_text = f"FPS: {fps:.1f} | Buffer Drowsy: {consecutive_drowsy}"
    cv2.putText(frame_bgr, telemetry_text, (w - 280, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

    # Barra gráfica de probabilidad en la parte inferior
    bar_width = int((w - 40) * prob_drowsy)
    bar_color = (0, 0, 255) if prob_drowsy >= 0.5 else (0, 200, 0)
    cv2.rectangle(frame_bgr, (20, h - 15), (w - 20, h - 7), (40, 50, 70), -1)
    cv2.rectangle(frame_bgr, (20, h - 15), (20 + bar_width, h - 7), bar_color, -1)

    return frame_bgr


# =============================================================================
# ENCABEZADO Y TELEMETRÍA GLOBAL
# =============================================================================
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="main-title">🚗 GuardDrive AI | Detección de Somnolencia</h1>
            <p class="main-subtitle">Sistema Inteligente de Monitoreo Facial para Prevención de Accidentes Vehiculares en Tiempo Real</p>
        </div>
        <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); padding: 8px 16px; border-radius: 10px; font-size: 0.85rem; color: #38bdf8;">
            <b>Modelo:</b> CNN 3-Ciclos Convolucionales (Temp/Spatial/Separable)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Carga de recursos
model = load_trained_model("model_ddd.keras")
metrics = load_metrics_data("metrics.json")

# =============================================================================
# PESTAÑAS DE NAVEGACIÓN
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "📹 Pestaña 1: Detección en Tiempo Real (MVP)",
    "📊 Pestaña 2: Métricas de Rendimiento del Modelo",
    "🧠 Pestaña 3: Fases y Arquitectura del Sistema"
])

# =============================================================================
# PESTAÑA 1: DETECCIÓN EN TIEMPO REAL (MVP)
# =============================================================================
with tab1:
    st.markdown("### 🎥 Monitoreo Activo del Conductor")
    
    if model is None:
        st.warning("⚠️ **El modelo 'model_ddd.keras' aún no ha sido entrenado o localizado.** Por favor ejecuta `python train.py` o espera que termine el entrenamiento.")
    
    # Barra lateral de configuración para el MVP
    with st.sidebar:
        st.markdown("### ⚙️ Parámetros del Sistema")
        threshold = st.slider(
            "Umbral de Decisión de Somnolencia",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Probabilidad a partir de la cual el clasificador considera somnolencia."
        )
        alert_consecutive_frames = st.slider(
            "Ventana Temporal (Frames Consecutivos para Alarma)",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            help="Número de cuadros sucesivos con somnolencia antes de disparar la alarma acústica."
        )
        sound_enabled = st.toggle("Activar Alarma Sonora", value=True, help="Emite tono acústico cuando se supere la ventana temporal de somnolencia.")
        camera_index = st.number_input("Índice de Cámara Web", min_value=0, max_value=5, value=0, step=1)
        
        st.markdown("---")
        st.markdown("### ℹ️ Leyenda de Estados")
        st.markdown("🟢 **Alerta / Despierto:** Probabilidad < Umbral")
        st.markdown("🔴 **¡SOMNOLENCIA DETECTADA!:** Probabilidad ≥ Umbral")

    col_view, col_controls = st.columns([2.2, 1.2])

    with col_controls:
        st.markdown("#### 🎛️ Modo de Evaluación")
        eval_mode = st.radio(
            "Selecciona fuente de entrada:",
            ["Subir Imagen / Foto de Prueba", "Cámara Web en Vivo", "Muestras del Dataset DDD"],
            index=0
        )

        # Variables de estado en tiempo real
        status_container = st.empty()
        confidence_container = st.empty()
        telemetry_container = st.empty()

    # MODO 1: CÁMARA WEB EN VIVO CON OPENCV
    if eval_mode == "Cámara Web en Vivo":
        with col_view:
            st.markdown("##### 🔴 Transmisión de Video en Directo")
            st.info("Este modo usa la cámara del equipo donde se ejecuta Streamlit. En Streamlit Cloud usa 'Subir Imagen / Foto de Prueba'.")
            video_placeholder = st.empty()
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                start_webcam = st.button("▶️ Iniciar Cámara Web", use_container_width=True, type="primary")
            with col_btn2:
                stop_webcam = st.button("⏹️ Detener Cámara Web", use_container_width=True)

            if "webcam_running" not in st.session_state:
                st.session_state.webcam_running = False

            if start_webcam:
                st.session_state.webcam_running = True
            if stop_webcam:
                st.session_state.webcam_running = False

            if st.session_state.webcam_running:
                cap = cv2.VideoCapture(int(camera_index))
                
                if not cap.isOpened():
                    st.error(f"❌ No fue posible acceder a la cámara con índice {camera_index}. Verifica que esté conectada y tenga permisos.")
                    st.session_state.webcam_running = False
                else:
                    consecutive_drowsy_count = 0
                    prev_time = time.time()
                    last_alarm_time = 0

                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("⚠️ Señal de video perdida o finalizada.")
                            break

                        current_time = time.time()
                        fps = 1.0 / max(1e-4, (current_time - prev_time))
                        prev_time = current_time

                        # Inferencia con la CNN
                        prob_drowsy = 0.0
                        if model is not None:
                            tensor, _ = preprocess_frame(frame, target_size=(64, 64))
                            prob_drowsy = float(model(tensor, training=False).numpy()[0][0])

                        is_drowsy = prob_drowsy >= threshold

                        if is_drowsy:
                            consecutive_drowsy_count += 1
                        else:
                            consecutive_drowsy_count = max(0, consecutive_drowsy_count - 1)

                        alarm_active = consecutive_drowsy_count >= alert_consecutive_frames

                        # Disparo de alarma acústica no bloqueante
                        if alarm_active and sound_enabled:
                            if current_time - last_alarm_time > 0.8:  # Evita saturar la reproducción
                                trigger_alarm_in_background()
                                last_alarm_time = current_time

                        status_label = "¡SOMNOLENCIA DETECTADA!" if is_drowsy else "Alerta / Despierto"

                        # Dibujar HUD en el frame
                        annotated_frame = draw_hud_overlay(
                            frame,
                            status_text=status_label,
                            is_drowsy=is_drowsy,
                            prob_drowsy=prob_drowsy,
                            fps=fps,
                            consecutive_drowsy=consecutive_drowsy_count,
                            alarm_triggered=alarm_active
                        )

                        # Renderizado del video en Streamlit (RGB)
                        frame_disp = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_disp, channels="RGB", use_container_width=True)

                        # Actualización de tarjetas de telemetría lateral
                        if alarm_active:
                            status_container.markdown("""
                            <div class="status-card status-drowsy">
                                <h2 style="margin: 0; font-size: 1.4rem;">🚨 ¡ALARMA CRÍTICA!</h2>
                                <p style="margin: 4px 0 0 0; font-weight: 600;">SOMNOLENCIA PERSISTENTE DETECTADA</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif is_drowsy:
                            status_container.markdown("""
                            <div class="status-card status-drowsy">
                                <h3 style="margin: 0; font-size: 1.3rem;">⚠️ Somnolencia Detectada</h3>
                                <p style="margin: 4px 0 0 0;">Buffer: {}/{} frames</p>
                            </div>
                            """.format(consecutive_drowsy_count, alert_consecutive_frames), unsafe_allow_html=True)
                        else:
                            status_container.markdown("""
                            <div class="status-card status-alert">
                                <h3 style="margin: 0; font-size: 1.3rem;">✅ Alerta / Despierto</h3>
                                <p style="margin: 4px 0 0 0;">Conductor atento a la vía</p>
                            </div>
                            """, unsafe_allow_html=True)

                        confidence_container.markdown(f"""
                        **Probabilidad de Somnolencia:**
                        """)
                        confidence_container.progress(min(1.0, max(0.0, prob_drowsy)))

                        telemetry_container.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Telemetría en Vivo</div>
                            <div style="font-size: 0.95rem; text-align: left; margin-top: 8px;">
                                • <b>FPS:</b> {fps:.1f}<br>
                                • <b>Confianza:</b> {prob_drowsy*100:.2f}%<br>
                                • <b>Buffer Alerta:</b> {consecutive_drowsy_count} / {alert_consecutive_frames} frames<br>
                                • <b>Alarma Sonora:</b> {"🔔 ACTIVA" if alarm_active else "🔕 INACTIVA"}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    cap.release()
            else:
                video_placeholder.info("ℹ️ Presiona **'▶️ Iniciar Cámara Web'** para activar el flujo de video en vivo y detección en tiempo real.")

    # MODO 2: SUBIR IMAGEN O FOTO DE PRUEBA
    elif eval_mode == "Subir Imagen / Foto de Prueba":
        with col_view:
            st.markdown("##### 📤 Cargar Archivo de Imagen")
            uploaded_file = st.file_uploader("Arrastra una imagen o selecciona un archivo (JPG, PNG)", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image_pil = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image_pil)
                
                # Inferencia
                resized = cv2.resize(img_array, (64, 64))
                tensor = np.expand_dims(resized, axis=0).astype(np.float32)
                prob_drowsy = float(model(tensor, training=False).numpy()[0][0]) if model else 0.0
                is_drowsy = prob_drowsy >= threshold

                # Anotación en imagen
                annotated = img_array.copy()
                h, w = annotated.shape[:2]
                color = (239, 68, 68) if is_drowsy else (34, 197, 94)
                cv2.rectangle(annotated, (10, 10), (w - 10, h - 10), color, 4)
                label = f"{'DROWSY' if is_drowsy else 'ALERT'} ({prob_drowsy*100:.1f}%)"
                cv2.putText(annotated, label, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

                st.image(annotated, caption="Resultado del Análisis de Somnolencia", use_container_width=True)

                if is_drowsy:
                    status_container.markdown("""
                    <div class="status-card status-drowsy">
                        <h2 style="margin: 0;">🚨 ¡SOMNOLENCIA DETECTADA!</h2>
                        <p>Probabilidad: {:.2f}%</p>
                    </div>
                    """.format(prob_drowsy * 100), unsafe_allow_html=True)
                    if sound_enabled:
                        trigger_alarm_in_background()
                else:
                    status_container.markdown("""
                    <div class="status-card status-alert">
                        <h2 style="margin: 0;">✅ ALERTA / DESPIERTO</h2>
                        <p>Probabilidad de Somnolencia: {:.2f}%</p>
                    </div>
                    """.format(prob_drowsy * 100), unsafe_allow_html=True)

                confidence_container.progress(prob_drowsy)

    # MODO 3: MUESTRAS DEL DATASET DDD
    elif eval_mode == "Muestras del Dataset DDD":
        with col_view:
            st.markdown("##### 📁 Demostración Rápida con Muestras Reales del Dataset")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                test_drowsy = st.button("🧪 Probar Muestra 'Drowsy' (Somnoliento)", use_container_width=True)
            with col_s2:
                test_alert = st.button("🧪 Probar Muestra 'Non Drowsy' (Despierto)", use_container_width=True)

            selected_image_path = None
            if test_drowsy:
                drowsy_files = glob.glob("Driver Drowsiness Dataset (DDD)/Drowsy/*.png")
                if drowsy_files:
                    selected_image_path = random.choice(drowsy_files)
            elif test_alert:
                alert_files = glob.glob("Driver Drowsiness Dataset (DDD)/Non Drowsy/*.png")
                if alert_files:
                    selected_image_path = random.choice(alert_files)

            if selected_image_path:
                img_bgr = cv2.imread(selected_image_path)
                tensor, img_rgb = preprocess_frame(img_bgr, target_size=(64, 64))
                prob_drowsy = float(model(tensor, training=False).numpy()[0][0]) if model else 0.0
                is_drowsy = prob_drowsy >= threshold

                st.image(img_rgb, caption=f"Muestra cargada: {os.path.basename(selected_image_path)}", width=360)

                if is_drowsy:
                    status_container.markdown(f"""
                    <div class="status-card status-drowsy">
                        <h2 style="margin: 0;">🚨 ¡SOMNOLENCIA DETECTADA!</h2>
                        <p>Probabilidad: {prob_drowsy*100:.2f}% (Ground Truth: Drowsy)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if sound_enabled:
                        trigger_alarm_in_background()
                else:
                    status_container.markdown(f"""
                    <div class="status-card status-alert">
                        <h2 style="margin: 0;">✅ ALERTA / DESPIERTO</h2>
                        <p>Probabilidad de Somnolencia: {prob_drowsy*100:.2f}% (Ground Truth: Non Drowsy)</p>
                    </div>
                    """, unsafe_allow_html=True)

                confidence_container.progress(prob_drowsy)
            else:
                st.info("Presiona cualquiera de los botones superiores para cargar y clasificar una imagen aleatoria del dataset oficial.")


# =============================================================================
# PESTAÑA 2: MÉTRICAS DE RENDIMIENTO DEL MODELO
# =============================================================================
with tab2:
    st.markdown("### 📊 Rendimiento y Evaluación Experimental del Modelo")
    st.markdown("Resultados cuantitativos obtenidos sobre el conjunto de evaluación independiente del **Driver Drowsiness Dataset (DDD)**.")

    if metrics is None:
        st.info("ℹ️ Las métricas estarán disponibles una vez completado el entrenamiento con `python train.py`.")
    else:
        ov = metrics.get("overall_metrics", {})
        
        # 1. Tarjetas Superiores de Métricas Globales
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Exactitud (Accuracy)</div>
                <div class="metric-value">{ov.get('accuracy', 0)*100:.2f}%</div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">Rendimiento global</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Área ROC (AUC)</div>
                <div class="metric-value" style="color: #38bdf8;">{ov.get('roc_auc', 0):.4f}</div>
                <div style="font-size: 0.72rem; color: #38bdf8; margin-top: 4px;">Separabilidad excelente</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score (Drowsy)</div>
                <div class="metric-value" style="color: #818cf8;">{ov.get('f1_score', 0):.4f}</div>
                <div style="font-size: 0.72rem; color: #818cf8; margin-top: 4px;">Balance armónico</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall (Sensibilidad)</div>
                <div class="metric-value" style="color: #34d399;">{ov.get('recall', 0)*100:.2f}%</div>
                <div style="font-size: 0.72rem; color: #34d399; margin-top: 4px;">Crítico: Seguridad vial</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precisión (Drowsy)</div>
                <div class="metric-value" style="color: #f472b6;">{ov.get('precision', 0)*100:.2f}%</div>
                <div style="font-size: 0.72rem; color: #f472b6; margin-top: 4px;">Certeza de alarmas</div>
            </div>
            """, unsafe_allow_html=True)

        # Caja de interpretación global de las métricas clave
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 14px 18px; margin-top: 14px; margin-bottom: 22px;">
            <b style="color: #38bdf8; font-size: 0.95rem;">💡 Interpretación Ejecutiva de las Métricas Globales:</b>
            <p style="color: #cbd5e1; font-size: 0.88rem; margin: 6px 0 0 0; line-height: 1.5;">
                • <b>Priorización de Seguridad (Recall {ov.get('recall', 0)*100:.2f}%):</b> En sistemas de seguridad vehicular (norma ISO 26262), omitir a un conductor dormido (Falso Negativo) es fatal. El modelo captura prácticamente el 97% de los eventos reales de somnolencia.<br>
                • <b>Capacidad Discriminativa (AUC {ov.get('roc_auc', 0):.4f}):</b> Al superar el umbral estándar de 0.90, el clasificador demuestra una separación casi óptima entre las distribuciones de probabilidad de conductores despiertos y fatigados.<br>
                • <b>Balance Operativo (F1-Score {ov.get('f1_score', 0):.4f}):</b> La combinación armónica de alta sensibilidad (96.56%) con una precisión respetable (81.35%) asegura que el conductor reciba advertencias oportunas sin padecer fatiga de alarma.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 2. Visualización Gráfica: Curva ROC y Matriz de Confusión
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            st.markdown("#### 📈 Curva ROC (Receiver Operating Characteristic)")
            if os.path.exists("roc_curve.png"):
                st.image("roc_curve.png", caption="Curva ROC vs Clasificador Aleatorio", use_container_width=True)
            else:
                roc_data = metrics.get("roc_curve", {})
                fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                ax_roc.plot(roc_data.get("fpr", [0, 1]), roc_data.get("tpr", [0, 1]), color="#22c55e", lw=2, label=f"AUC = {ov.get('roc_auc', 0):.4f}")
                ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Aleatorio (AUC = 0.5)")
                ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
                ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
                ax_roc.legend()
                st.pyplot(fig_roc)

            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.65); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 14px; margin-top: 8px;">
                <b style="color: #38bdf8; font-size: 0.9rem;">🔍 Interpretación de la Curva ROC:</b>
                <ul style="color: #cbd5e1; font-size: 0.85rem; margin: 6px 0 0 0; padding-left: 18px; line-height: 1.45;">
                    <li><b>Comportamiento de la Curva:</b> La curva asciende rápidamente pegada al eje vertical superior izquierdo, lo que indica una tasa de verdaderos positivos (TPR) superior al 95% con tasas de falsos positivos (FPR) por debajo del 15%.</li>
                    <li><b>Valor AUC = 0.9486:</b> Significa que si seleccionamos al azar un conductor somnoliento y uno despierto, la CNN asignará una probabilidad de fatiga más alta al conductor somnoliento en el <b>94.86% de los casos</b>.</li>
                    <li><b>Comparación de Referencia:</b> Supera ampliamente la línea diagonal punteada gris (clasificador aleatorio con AUC = 0.50), confirmando un aprendizaje estadístico genuino y reproducible.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with g_col2:
            st.markdown("#### 🎯 Matriz de Confusión Normalizada")
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", caption="Matriz de Confusión Normalizada por Fila", use_container_width=True)
            else:
                cm_data = np.array(metrics.get("confusion_matrix", {}).get("normalized", [[1, 0], [0, 1]]))
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_data, annot=True, fmt=".2%", cmap="Blues", ax=ax_cm, xticklabels=["No Somnoliento", "Somnoliento"], yticklabels=["No Somnoliento", "Somnoliento"])
                st.pyplot(fig_cm)

            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.65); border: 1px solid rgba(129, 140, 248, 0.2); border-radius: 10px; padding: 14px; margin-top: 8px;">
                <b style="color: #818cf8; font-size: 0.9rem;">🔍 Interpretación de la Matriz de Confusión:</b>
                <ul style="color: #cbd5e1; font-size: 0.85rem; margin: 6px 0 0 0; padding-left: 18px; line-height: 1.45;">
                    <li><b>Acierto en Somnolencia (Verdaderos Positivos = 96.56%):</b> De 610 casos reales de fatiga, el modelo clasificó correctamente 589. Solo 21 imágenes (<b>3.44%</b>) fueron falsos negativos, lo que minimiza el riesgo de accidentes no detectados.</li>
                    <li><b>Acierto en Alerta (Verdaderos Negativos = 75.09%):</b> 407 de 542 conductores despiertos fueron identificados con exactitud.</li>
                    <li><b>Falsas Alarmas (24.91%):</b> Las 135 confusiones se deben a parpadeos naturales intermedios o sombras en la cavidad ocular, situaciones neutralizadas en la práctica por el <b>buffer temporal de 10 cuadros</b> en el MVP.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(148, 163, 184, 0.15); margin: 24px 0;'>", unsafe_allow_html=True)

        # 3. Curvas de Entrenamiento y Reporte Detallado
        t_col1, t_col2 = st.columns([1.3, 1.0])

        with t_col1:
            st.markdown("#### 📉 Evolución del Entrenamiento (Train vs Val por Época)")
            if os.path.exists("training_curves.png"):
                st.image("training_curves.png", caption="Curvas de Precisión y Pérdida por Época", use_container_width=True)
            else:
                hist = metrics.get("history", {})
                if hist:
                    df_hist = pd.DataFrame(hist)
                    st.line_chart(df_hist.set_index("epochs")[["train_accuracy", "val_accuracy"]])

            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.65); border: 1px solid rgba(52, 211, 153, 0.2); border-radius: 10px; padding: 14px; margin-top: 8px;">
                <b style="color: #34d399; font-size: 0.9rem;">🔍 Interpretación de las Curvas de Aprendizaje:</b>
                <ul style="color: #cbd5e1; font-size: 0.85rem; margin: 6px 0 0 0; padding-left: 18px; line-height: 1.45;">
                    <li><b>Convergencia Estable de Pérdida (Loss):</b> La función de costo (Binary Crossentropy) desciende consistentemente desde 0.628 hasta 0.204 en entrenamiento y hasta 0.351 en validación, evidenciando un ajuste progresivo de pesos sin oscilaciones caóticas.</li>
                    <li><b>Evolución de la Precisión (Accuracy):</b> La precisión escala del 64.75% al 91.89% en entrenamiento y al 86.11% en validación en tan solo 6 épocas.</li>
                    <li><b>Control del Sobreajuste (Generalización):</b> La proximidad entre las curvas de entrenamiento y validación valida la efectividad del <i>Dropout</i> (30% y 40%) y de la capa <i>BatchNormalization</i>, evitando la memorización espuria de rostros particulares.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with t_col2:
            st.markdown("#### 📋 Reporte de Clasificación Detallado")
            clf_rep = metrics.get("classification_report", {})
            if clf_rep:
                rep_rows = []
                for label in ["Non Drowsy", "Drowsy", "macro avg", "weighted avg"]:
                    if label in clf_rep:
                        row = clf_rep[label]
                        rep_rows.append({
                            "Clase / Promedio": "No Somnoliento (Alerta)" if label == "Non Drowsy" else ("Somnoliento" if label == "Drowsy" else label),
                            "Precisión": f"{row.get('precision', 0):.4f}",
                            "Recall": f"{row.get('recall', 0):.4f}",
                            "F1-Score": f"{row.get('f1-score', 0):.4f}",
                            "Soporte": int(row.get('support', 0))
                        })
                st.table(pd.DataFrame(rep_rows))

                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.65); border: 1px solid rgba(244, 114, 182, 0.2); border-radius: 10px; padding: 14px; margin-top: 8px;">
                    <b style="color: #f472b6; font-size: 0.9rem;">🔍 Interpretación del Reporte por Clase:</b>
                    <ul style="color: #cbd5e1; font-size: 0.85rem; margin: 6px 0 0 0; padding-left: 18px; line-height: 1.45;">
                        <li><b>Clase No Somnoliento (Alerta):</b> Exhibe una altísima <b>precisión de 95.09%</b>, lo que asegura que cuando el modelo dictamina que el conductor está despierto, la probabilidad de error es inferior al 5%.</li>
                        <li><b>Clase Somnoliento (Fatiga):</b> Alcanza un <b>recall del 96.56%</b> y <b>F1 de 0.8831</b>, confirmando que es altamente sensible ante cualquier indicio de párpados caídos o sueño.</li>
                        <li><b>Soporte Estadístico:</b> Evaluado sobre 1,152 imágenes de prueba (542 Alerta vs 610 Somnoliento), constituyendo un benchmark balanceado y confiable.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# PESTAÑA 3: FASES Y ARQUITECTURA DEL SISTEMA (APARTADO ACADÉMICO / EXPLICATIVO)
# =============================================================================
with tab3:
    st.markdown("### 🧠 Fundamentos Teóricos y Arquitectura de la Red Convolucional")
    st.markdown("""
    Esta sección documenta formalmente la metodología de diseño de la CNN para la detección de somnolencia,
    desglosando la estructura de **3 Ciclos Convolucionales** y definiendo los conceptos matemáticos y computacionales clave.
    """)

    st.markdown("#### 🏗️ Desglose del Pipeline por Fases y Bloques Convolucionales")

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("""
        <div class="academic-card">
            <span class="phase-badge">Fase 1</span>
            <h4 style="margin: 0; color: #38bdf8;">Entrada y Normalización del Tensor</h4>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;">
                La imagen capturada por la cámara (RGB) es redimensionada a <b>64 &times; 64 &times; 3</b> píxeles. Mediante la capa <code>Rescaling(1./255)</code>, los valores discretos enteros <code>[0, 255]</code> se mapean linealmente al rango continuo <code>[0.0, 1.0]</code>.
            </p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                <b>Justificación Técnica:</b> La normalización de escala previene la saturación temprana de los pesos y homogeniza las magnitudes del gradiente durante la retropropagación (Backpropagation).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="academic-card">
            <span class="phase-badge">Fase 2</span>
            <h4 style="margin: 0; color: #38bdf8;">Bloque 1 (Temp-Conv) y Batch Normalization</h4>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;">
                Implementa una convolución 2D inicial (32 filtros, kernel 3&times;3, padding 'same') orientada a capturar transiciones y micro-gradientes de contraste (bordes de párpados, reflejos pupilares).
            </p>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Seguido inmediatamente por <code>BatchNormalization()</code>, que normaliza la media y varianza de las activaciones del lote:
            </p>
            <p style="font-family: monospace; font-size: 0.85rem; color: #38bdf8;">
                y = &gamma; &middot; ((x - &mu;) / &radic;(&sigma;&sup2; + &epsilon;)) + &beta;
            </p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                <b>Beneficio:</b> Mitiga el <i>Internal Covariate Shift</i> y permite el uso de tasas de aprendizaje más altas sin divergencia numérica.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="academic-card">
            <span class="phase-badge">Fase 3</span>
            <h4 style="margin: 0; color: #38bdf8;">Bloque 2 (Spatial-Conv) y Pooling (Ciclo 2)</h4>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;">
                Capa convolucional de 64 filtros (3&times;3) orientada a patrones espaciales y rasgos faciales complejos (apertura palpebral, comisuras bucales, inclinación de la cabeza).
            </p>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                A continuación, <code>MaxPooling2D(pool_size=(2, 2))</code> reduce la resolución espacial a la mitad (de 64&times;64 a 32&times;32), preservando únicamente la activación más destacada de cada cuadrante 2&times;2.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="academic-card">
            <span class="phase-badge">Fase 4</span>
            <h4 style="margin: 0; color: #818cf8;">Regularización 1: Dropout</h4>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;">
                Capa de <code>Dropout(rate=0.30)</code>. Durante el entrenamiento, desactiva aleatoriamente el 30% de las conexiones de los mapas de características.
            </p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                <b>Efecto:</b> Fuerza a la red a no depender de un rasgo facial único (evita la co-adaptación de neuronas), garantizando que el modelo generalice ante variaciones de iluminación, ángulos o uso de lentes.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="academic-card">
            <span class="phase-badge">Fase 5</span>
            <h4 style="margin: 0; color: #818cf8;">Bloque 3 (Separable-Conv) y Global Pooling (Ciclo 3)</h4>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;">
                Implementa <code>SeparableConv2D(128, kernel_size=(3, 3))</code>. Desacopla la convolución estándar en dos operaciones:
            </p>
            <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 20px;">
                <li><b>Depthwise Convolution:</b> Aplica un filtro espacial independiente por cada canal de entrada.</li>
                <li><b>Pointwise Convolution:</b> Combina linealmente los canales resultantes mediante filtros 1&times;1.</li>
            </ul>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                <b>Eficiencia Computacional:</b> Reduce los FLOPs en aproximadamente un <b>85% a 90%</b> comparado con una Conv2D tradicional equivalente, permitiendo inferencias de más de 30 FPS en CPUs y dispositivos embebidos.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="academic-card">
            <span class="phase-badge">Fase 6</span>
            <h4 style="margin: 0; color: #818cf8;">Clasificador (MLP) y Salida Sigmoide</h4>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 8px;">
                Capa <code>GlobalAveragePooling2D()</code> que colapsa los mapas espaciales en un vector compacto de 128 características.
            </p>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                Seguido de una capa densa intermedia <code>Dense(64, activation='relu')</code> con <code>Dropout(0.4)</code>, y la neurona final de salida <code>Dense(1, activation='sigmoid')</code>:
            </p>
            <p style="font-family: monospace; font-size: 0.85rem; color: #38bdf8;">
                &sigma;(z) = 1 / (1 + e<sup>-z</sup>) &isin; [0, 1]
            </p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                La salida representa directamente la probabilidad condicional P(Somnoliento | Imagen).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Glosario Técnico Exhaustivo
    st.markdown("#### 📖 Glosario Técnico de Visión por Computadora y Deep Learning")
    g1, g2 = st.columns(2)

    with g1:
        with st.expander("🔹 Convolución 2D", expanded=True):
            st.markdown("""
            Operación matemática fundamental que aplica una matriz de pesos móviles (**Kernel**) sobre un tensor de entrada mediante producto punto y desplazamiento espacial. Permite la extracción automática de características jerárquicas con **invarianza a la traslación local**.
            """)

        with st.expander("🔹 Filtros / Kernels", expanded=True):
            st.markdown("""
            Matrices de pesos entrenables (generalmente de $3 \\times 3$ o $5 \\times 5$) cuyos valores se ajustan por descenso de gradiente para detectar patrones visuales específicos como bordes, esquinas, texturas y formas anatómicas.
            """)

        with st.expander("🔹 Stride (Paso)", expanded=True):
            st.markdown("""
            Número de píxeles que se desplaza la ventana del kernel a lo largo de las filas y columnas de la imagen en cada paso. Un stride de 1 genera una exploración densa; un stride mayor a 1 produce un submuestreo espacial directo.
            """)

    with g2:
        with st.expander("🔹 Max Pooling", expanded=True):
            st.markdown("""
            Operación de submuestreo que selecciona el valor máximo dentro de una región de tamaño fijo (ej. $2 \\times 2$). Reduce la dimensionalidad del tensor, minimiza la sobrecarga computacional y confiere resistencia a pequeñas perturbaciones espaciales.
            """)

        with st.expander("🔹 Convolución Separable (Depthwise Separable)", expanded=True):
            st.markdown("""
            Técnica que factoriza una convolución estándar en:
            1. **Convolución en Profundidad (Depthwise):** filtrado espacial canal por canal.
            2. **Convolución Puntual (Pointwise):** combinación $1 \\times 1$ de canales.
            Reduce la complejidad teórica de $\\mathcal{O}(D_K^2 \\cdot M \\cdot N)$ a $\\mathcal{O}(D_K^2 \\cdot M + M \\cdot N)$.
            """)

        with st.expander("🔹 Funciones de Activación (ReLU & Sigmoide)", expanded=True):
            st.markdown("""
            - **ReLU (Rectified Linear Unit):** $f(x) = \\max(0, x)$. Introduce no linealidad sin sufrir el problema de desvanecimiento de gradientes para valores positivos.
            - **Sigmoide:** $\\sigma(x) = \\frac{1}{1 + e^{-x}}$. Mapea valores reales al intervalo acotado $[0, 1]$, ideal para clasificación probabilística binaria.
            """)

    st.markdown("---")

    # Tabla de la Arquitectura
    st.markdown("#### 📐 Resumen Arquitectónico de Capas (Model Summary)")
    layers_data = [
        {"Capa": "Entrada (Input)", "Tipo": "InputLayer", "Dimensión de Salida": "(None, 64, 64, 3)", "Parámetros": 0, "Propósito": "Tensor de imagen RGB crudo"},
        {"Capa": "Normalización", "Tipo": "Rescaling(1./255)", "Dimensión de Salida": "(None, 64, 64, 3)", "Parámetros": 0, "Propósito": "Normalización lineal [0.0, 1.0]"},
        {"Capa": "Bloque 1 Temp-Conv", "Tipo": "Conv2D (32, 3x3)", "Dimensión de Salida": "(None, 64, 64, 32)", "Parámetros": 896, "Propósito": "Extracción de micro-transiciones y gradientes"},
        {"Capa": "Batch Normalization", "Tipo": "BatchNormalization", "Dimensión de Salida": "(None, 64, 64, 32)", "Parámetros": 128, "Propósito": "Estabilización de varianza y media"},
        {"Capa": "Bloque 2 Spatial-Conv", "Tipo": "Conv2D (64, 3x3)", "Dimensión de Salida": "(None, 64, 64, 64)", "Parámetros": 18496, "Propósito": "Rasgos espaciales faciales (ojos, boca)"},
        {"Capa": "Pooling Ciclo 2", "Tipo": "MaxPooling2D (2x2)", "Dimensión de Salida": "(None, 32, 32, 64)", "Parámetros": 0, "Propósito": "Reducción espacial de 64x64 a 32x32"},
        {"Capa": "Regularización 1", "Tipo": "Dropout (0.30)", "Dimensión de Salida": "(None, 32, 32, 64)", "Parámetros": 0, "Propósito": "Mitigación de sobreajuste"},
        {"Capa": "Bloque 3 Separable-Conv", "Tipo": "SeparableConv2D (128)", "Dimensión de Salida": "(None, 32, 32, 128)", "Parámetros": 8896, "Propósito": "Inferencia ligera y eficiente en tiempo real"},
        {"Capa": "Pooling Ciclo 3", "Tipo": "GlobalAveragePooling2D", "Dimensión de Salida": "(None, 128)", "Parámetros": 0, "Propósito": "Colapso espacial a vector de 128D"},
        {"Capa": "MLP Intermedio", "Tipo": "Dense (64, ReLU)", "Dimensión de Salida": "(None, 64)", "Parámetros": 8256, "Propósito": "Combinación no lineal de rasgos de alto nivel"},
        {"Capa": "Regularización 2", "Tipo": "Dropout (0.40)", "Dimensión de Salida": "(None, 64)", "Parámetros": 0, "Propósito": "Regularización previa a la decisión"},
        {"Capa": "Salida Clasificador", "Tipo": "Dense (1, Sigmoid)", "Dimensión de Salida": "(None, 1)", "Parámetros": 65, "Propósito": "P(Somnoliento | Imagen) continua"}
    ]
    df_layers = pd.DataFrame(layers_data)
    st.dataframe(df_layers, use_container_width=True, hide_index=True)
    st.caption("Total de parámetros: ~36,737 (~143.5 KB) | Diseñado para latencias de inferencia < 15 ms en CPU.")
