# 🚗 GuardDrive AI - Detección de Somnolencia en Conductores en Tiempo Real

Sistema integral en Python para el monitoreo y detección de fatiga/somnolencia vehicular en tiempo real mediante Deep Learning y Visión por Computadora, desplegado en una aplicación web interactiva desarrollada con **Streamlit**.

---

## 📁 Estructura del Proyecto

```text
sprint2/
│
├── Driver Drowsiness Dataset (DDD)/   # Dataset con 41,793 imágenes (227x227)
│   ├── Drowsy/                        # 22,348 imágenes (Clase 1: Somnoliento)
│   └── Non Drowsy/                    # 19,445 imágenes (Clase 0: Alerta / Despierto)
│
├── train.py                           # Pipeline de carga, entrenamiento, evaluación y exportación
├── app.py                             # Aplicación web Streamlit + WebRTC (inferencia en servidor)
├── webrtc_live.py                     # Estado live thread-safe y configuración ICE/STUN/TURN
├── browser_alarm.py                   # Alarma Web Audio ejecutada en el navegador
├── tools/
│   └── generate_cert.py               # Generador de certificado HTTPS autofirmado para LAN
├── requirements.txt                   # Dependencias oficiales del entorno
├── packages.txt                       # Dependencias APT para Streamlit Cloud (OpenCV)
├── .streamlit/
│   └── secrets.toml.example           # Plantilla de TURN para Streamlit Cloud
├── README.md                          # Documentación técnica del proyecto
├── INFORME_SISTEMA.md                 # Informe académico del sistema
│
├── model_ddd.keras                    # Modelo entrenado Keras de 3 Ciclos Convolucionales
├── metrics.json                       # Métricas cuantitativas (ROC, AUC, Matriz de Confusión, F1)
├── roc_curve.png                      # Gráfica de la Curva ROC con AUC
├── confusion_matrix.png               # Gráfica de la Matriz de Confusión normalizada
├── training_curves.png                # Gráfica de curvas de Loss y Accuracy por época
├── alarm.wav                          # Audio embebido (base64) para la alarma del navegador
│
├── certs/                             # Certificados HTTPS generados localmente (ignorado por git)
└── backups/                           # Copias de seguridad y logs de entrenamiento (ignorado por git)
```

---

## 🧠 Arquitectura de la Red Neuronal Convolucional (3 Ciclos Convolucionales)

La arquitectura de la red neuronal convolucional se implementa en la función `build_cnn_architecture()` del archivo [`train.py`](file:///d:/sprint2/train.py#L74-L138) (líneas 74 a 138). A continuación se detalla la ubicación exacta por **número de línea de código** para cada fase, bloque y ciclo convolucional:

| Etapa / Bloque | Capa Keras / Operación | Archivo | Líneas de Código | Propósito Técnico |
| :--- | :--- | :---: | :---: | :--- |
| **Definición Función** | `def build_cnn_architecture()` | `train.py` | [Líneas 74–85](file:///d:/sprint2/train.py#L74-L85) | Encapsulación modular de la CNN |
| **1. Entrada y Reescalado** | `layers.Input(shape=(64, 64, 3))` + `Rescaling(1./255)` | `train.py` | [Líneas 87–89](file:///d:/sprint2/train.py#L87-L89) | Normalización continua de píxeles a $[0.0, 1.0]$ |
| **2. Bloque 1 (Temp-Conv)** | `layers.Conv2D(32, (3, 3), padding='same', ReLU)` | `train.py` | [Líneas 91–98](file:///d:/sprint2/train.py#L91-L98) | Extracción de micro-transiciones y gradientes locales |
| **3. Normalización 1** | `layers.BatchNormalization()` | `train.py` | [Líneas 100–101](file:///d:/sprint2/train.py#L100-L101) | Estabilización del flujo de gradientes (Internal Covariate Shift) |
| **4. Bloque 2 (Spatial-Conv)** | `layers.Conv2D(64, (3, 3), padding='same', ReLU)` | `train.py` | [Líneas 103–110](file:///d:/sprint2/train.py#L103-L110) | Detección de rasgos espaciales faciales (ojos, boca, postura) |
| **5. Activación y Pooling (Ciclo 2)** | `layers.MaxPooling2D(pool_size=(2, 2))` | `train.py` | [Líneas 112–113](file:///d:/sprint2/train.py#L112-L113) | Reducción dimensional espacial ($64 \times 64 \to 32 \times 32$) |
| **6. Regularización 1** | `layers.Dropout(0.30)` | `train.py` | [Líneas 115–116](file:///d:/sprint2/train.py#L115-L116) | Prevención de co-adaptación de neuronas y sobreajuste |
| **7. Bloque 3 (Separable-Conv)** | `layers.SeparableConv2D(128, (3, 3), padding='same', ReLU)` | `train.py` | [Líneas 118–126](file:///d:/sprint2/train.py#L118-L126) | Convolución Depthwise + Pointwise (~88% menos FLOPs) |
| **8. Activación y Pooling (Ciclo 3)** | `layers.GlobalAveragePooling2D()` | `train.py` | [Líneas 128–129](file:///d:/sprint2/train.py#L128-L129) | Colapso de mapas espaciales a vector denso de 128D |
| **9. Clasificador (MLP) y Sigmoide** | `Dense(64, ReLU)` + `Dropout(0.40)` + `Dense(1, Sigmoid)` | `train.py` | [Líneas 131–134](file:///d:/sprint2/train.py#L131-L134) | Mapeo probabilístico final: $P(\text{Somnoliento} \mid X) \in [0, 1]$ |

### Detalle Específico por Línea en [`train.py`](file:///d:/sprint2/train.py):

1. **Entrada y Reescalado:** [Líneas 87–89](file:///d:/sprint2/train.py#L87-L89)
   - Línea 88: `layers.Input(shape=input_shape, name="input_tensor")`
   - Línea 89: `layers.Rescaling(1.0 / 255.0, name="rescaling_normalization")`
2. **Bloque 1 (Temp-Conv):** [Líneas 91–98](file:///d:/sprint2/train.py#L91-L98)
   - Líneas 92–98: `layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", name="block1_temp_conv")`
3. **Normalización 1:** [Líneas 100–101](file:///d:/sprint2/train.py#L100-L101)
   - Línea 101: `layers.BatchNormalization(name="batch_normalization_1")`
4. **Bloque 2 (Spatial-Conv):** [Líneas 103–110](file:///d:/sprint2/train.py#L103-L110)
   - Líneas 104–110: `layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="block2_spatial_conv")`
5. **Activación y Pooling (Ciclo 2):** [Líneas 112–113](file:///d:/sprint2/train.py#L112-L113)
   - Línea 113: `layers.MaxPooling2D(pool_size=(2, 2), name="pooling_cycle2_maxpool")`
6. **Regularización 1:** [Líneas 115–116](file:///d:/sprint2/train.py#L115-L116)
   - Línea 116: `layers.Dropout(0.3, name="dropout_regularization_1")`
7. **Bloque 3 (Separable-Conv):** [Líneas 118–126](file:///d:/sprint2/train.py#L118-L126)
   - Líneas 120–126: `layers.SeparableConv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", name="block3_separable_conv")`
8. **Activación y Pooling (Ciclo 3):** [Líneas 128–129](file:///d:/sprint2/train.py#L128-L129)
   - Línea 129: `layers.GlobalAveragePooling2D(name="pooling_cycle3_global_avg")`
9. **Clasificador (MLP) y Salida Sigmoide:** [Líneas 131–134](file:///d:/sprint2/train.py#L131-L134)
   - Línea 132: `layers.Dense(64, activation="relu", name="classifier_dense_intermediate")`
   - Línea 133: `layers.Dropout(0.4, name="classifier_dropout")`
   - Línea 134: `layers.Dense(1, activation="sigmoid", name="classifier_output_sigmoid")`

> **Nota de Implementación en [`app.py`](file:///d:/sprint2/app.py):**
> - La carga en caché del modelo se realiza en `load_trained_model()` ([Líneas 333–345](file:///d:/sprint2/app.py#L333-L345)).
> - El preprocesamiento del tensor se ejecuta en `preprocess_frame()` ([Líneas 359–370](file:///d:/sprint2/app.py#L359-L370)).
> - La inferencia thread-safe se centraliza en `predict_prob()` ([Líneas 289–294](file:///d:/sprint2/app.py#L289-L294)).
> - El callback frame a frame de WebRTC vive en `build_video_callback()` ([Líneas 297–329](file:///d:/sprint2/app.py#L297-L329)).
> - La configuración ICE/STUN/TURN se construye en `build_rtc_configuration()` de [`webrtc_live.py`](file:///d:/sprint2/webrtc_live.py).
> - La telemetría en vivo se refresca con `render_live_panel()` ([Líneas 438–500](file:///d:/sprint2/app.py#L438-L500)).
> - El stream WebRTC se declara en la Pestaña 1 ([Línea 674](file:///d:/sprint2/app.py#L674)).
> - La tabla y desglose académico se renderiza en la Pestaña 3 ([Línea 1010](file:///d:/sprint2/app.py#L1010)).

**Total de parámetros:** ~36,737 (~143.5 KB en disco, modelo final ~495 KB con metadatos Keras).

---

## 🚀 Instalación y Ejecución

### 1. Instalación de Dependencias
Abre una terminal en la carpeta `sprint 2/` e instala los paquetes necesarios:

```bash
pip install -r requirements.txt
```

> WebRTC requiere `streamlit-webrtc`, `aiortc`, `av` y `cryptography`. Las versiones compatibles ya están fijadas en `requirements.txt` (`streamlit-webrtc==0.79.0`, `aiortc==1.15.0`, `av>=15.1.0,<18.0.0`).

### 2. Entrenamiento y Evaluación del Modelo (`train.py`)
Para reentrenar el modelo o generar nuevas métricas:

```bash
# Entrenamiento completo (recomendado; ~20 min en CPU):
python train.py --epochs 15 --batch_size 64 --max_samples 0

# Entrenamiento ágil para pruebas (subconjunto balanceado):
python train.py --epochs 8 --batch_size 64 --max_samples 9600
```

Durante el entrenamiento se generan automáticamente:
- `model_ddd.keras` (pesos y arquitectura serializada)
- `metrics.json` (métricas cuantitativas completas)
- `roc_curve.png`, `confusion_matrix.png` y `training_curves.png`

#### Resultados del último reentrenamiento (dataset completo)

Evaluación sobre **6,265 imágenes de prueba** (2,899 `Non Drowsy` + 3,366 `Drowsy`):

| Métrica | Valor |
| :--- | ---: |
| Accuracy | **99.60%** |
| Precision (Drowsy) | **99.38%** |
| Recall (Drowsy) | **99.88%** |
| F1-Score (Drowsy) | **0.9963** |
| AUC-ROC | **1.0000** |
| Falsos negativos | **4** de 3,366 |
| Tiempo de entrenamiento | **1,193.73 s** (15 épocas, CPU) |

> **Nota metodológica:** el split es aleatorio sobre el dataset. Como el DDD contiene frames de video, pueden existir muestras casi duplicadas entre train y test. Para certificar la generalización real se requiere una validación externa con sujetos, cámaras e iluminación no vistos.

### 3. Lanzamiento de la Aplicación Web (`app.py`)

#### Opción A: uso local por HTTP
```bash
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`. La cámara funciona en `localhost` porque el navegador lo considera un contexto seguro.

#### Opción B: celular o PC en la misma WiFi (HTTPS obligatorio)
Los navegadores móviles solo permiten acceder a la cámara en **HTTPS** o `localhost`. Genera un certificado autofirmado y arranca Streamlit con SSL:

```bash
python tools/generate_cert.py

streamlit run app.py ^
  --server.address 0.0.0.0 ^
  --server.port 8501 ^
  --server.sslCertFile certs/cert.pem ^
  --server.sslKeyFile certs/key.pem
```

En PowerShell usa el acento grave para dividir líneas, o ejecuta todo en una sola línea:

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.sslCertFile certs/cert.pem --server.sslKeyFile certs/key.pem
```

1. Abre en el celular la URL que muestra la barra lateral, por ejemplo `https://192.168.0.102:8501`.
2. Acepta la advertencia de certificado autofirmado (**Avanzado → Continuar**).
3. Pulsa **START** en el reproductor WebRTC y permite la cámara.
4. Autoriza a Python/Streamlit en el Firewall de Windows para redes **privadas** (TCP 8501 y tráfico UDP de WebRTC).

> El certificado autofirmado es solo para pruebas en LAN. No expongas esta app directamente a internet sin un proxy TLS y autenticación. Si la IP del equipo cambia (DHCP), regenera el certificado con `python tools/generate_cert.py --force`.

#### Modos de red y funcionamiento sin internet

| Modo | ICE servers | Cuándo usarlo |
| :--- | :--- | :--- |
| **LAN / misma WiFi** (por defecto en local) | Solo candidatos host; no contacta STUN/TURN | Celular y PC en la misma WiFi, incluso sin salida a internet |
| **Internet / NAT** (por defecto en la nube) | STUN + TURN configurado o público de respaldo | Redes distintas, NAT restrictivo, aislamiento de clientes o despliegue en la nube |

La app no carga fuentes, CDN ni recursos externos: usa tipografías del sistema y embebe `alarm.wav` en base64. La alarma se reproduce con **Web Audio API en el navegador**, nunca en el servidor.

Para redes que requieren TURN, define las credenciales por variables de entorno o por `st.secrets` (no las escribas en el código):

```powershell
$env:TURN_URL = "turn:tu-servidor.example.com:3478"
$env:TURN_USERNAME = "usuario"
$env:TURN_PASSWORD = "clave-segura"
# Opcional: STUN propio
$env:STUN_URL = "stun:stun.l.google.com:19302"
```

En modo **Internet** puedes activar el TURN público de respaldo desde la interfaz. Es *best-effort* y depende de un servicio externo, por lo que no garantiza disponibilidad.

### 4. Despliegue en Streamlit Community Cloud

1. Sube el repositorio a GitHub. El modelo, métricas, gráficas, `alarm.wav`, `requirements.txt`, `packages.txt` y `.streamlit/secrets.toml.example` deben estar versionados.
2. En [share.streamlit.io](https://share.streamlit.io), crea una app y selecciona el repositorio, la rama y `app.py`.
3. En **Advanced settings**, selecciona **Python 3.12**. Streamlit Cloud ignora `runtime.txt`; la versión se elige en ese diálogo.
4. En **Secrets**, pega el contenido de `.streamlit/secrets.toml.example` con tus credenciales TURN reales. Sin TURN, la app activa el TURN público de respaldo en modo nube, que puede fallar.
5. Despliega y abre la URL `https://<tu-app>.streamlit.app` desde el celular.

Notas para la nube:

- **No subas** `Driver Drowsiness Dataset (DDD)/`, `certs/` ni `backups/`; ya están en `.gitignore`. El modo en vivo y la carga de imágenes funcionan sin el dataset. La pestaña de muestras DDD solo funciona si el dataset está presente localmente.
- **No uses** `--server.sslCertFile` ni `tools/generate_cert.py` en la nube: Streamlit Cloud termina TLS y provee HTTPS automáticamente.
- **WebRTC remoto:** la app detecta el host público y selecciona *Internet / NAT*; si no hay TURN propio, deja activado el TURN público de respaldo. Para una demo estable, configura Metered, Twilio o un coturn propio.
- **CORS/XSRF:** no los desactives. `streamlit-webrtc` 0.79 no los necesita y Streamlit Cloud mantiene la seguridad por defecto.
- **Recursos:** el plan gratuito tiene 1 GB de RAM. TensorFlow CPU y una sesión WebRTC pueden ser justos; usa 480×360 o 320×240 y una sola sesión durante la demo.

#### Solución de problemas rápidos
- **La cámara no abre en el celular:** la URL debe ser `https://`; acepta el certificado y vuelve a pulsar START.
- **WebRTC se queda en “connecting”:** verifica que ambos equipos estén en la misma WiFi y que el Firewall permita Python. Si la red aísla clientes, usa TURN. `streamlit-webrtc` 0.79 no requiere desactivar CORS/XSRF; no reduzcas la seguridad de Streamlit.
- **No suena la alarma:** haz al menos un clic/toque en la página para desbloquear el audio del navegador y confirma el toggle **Activar Alarma Sonora**.
- **No aparece la IP correcta:** la barra lateral lista la IP principal y otras interfaces; prueba la que corresponda a tu WiFi. Si la IP cambió, regenera el certificado con `python tools/generate_cert.py --force`.
- **En Streamlit Cloud falla la instalación de TensorFlow:** verifica que la app esté usando **Python 3.12** en Advanced settings; no dependas de `runtime.txt`.
- **En Streamlit Cloud WebRTC queda en “connecting”:** configura `TURN_URL`, `TURN_USERNAME` y `TURN_PASSWORD` en Secrets. STUN por sí solo no atraviesa todos los NAT móviles.
- **La pestaña del dataset está vacía en la nube:** es esperado; el dataset no se versiona. Usa el modo WebRTC o sube una imagen local.

---

## 🖥️ Módulos de la Aplicación Web (Streamlit)

### Pestaña 1: Detección en Tiempo Real (MVP)
- **Video en vivo por WebRTC:** La cámara del celular o navegador envía el stream al servidor con `streamlit-webrtc`; la CNN infiere frame a frame y devuelve el video anotado. Ya no se usa `cv2.VideoCapture` para la cámara local.
- **HUD Integrado:** Superpone telemetría en vivo, FPS estimado, probabilidad, buffer temporal y estado ("Alerta / Despierto" o "¡SOMNOLENCIA DETECTADA!").
- **Alarma en el navegador:** Si la somnolencia se sostiene durante $N$ cuadros consecutivos (slider configurable), un marcador oculto activa la alarma con **Web Audio API** en el navegador del usuario. El servidor no reproduce audio (`winsound`/`pygame` eliminados).
- **Selector de red:** modo **LAN / sin internet** (solo candidatos host) o **Internet / NAT** (STUN/TURN), cámara frontal/trasera, resolución y FPS objetivo.
- **Botón de prueba:** permite verificar la alarma en el navegador antes de iniciar el monitoreo.
- **Modos de Evaluación Alternativos:** carga de imágenes locales (`.jpg`, `.png`) y muestras aleatorias del dataset DDD, también con alarma en navegador.

### Pestaña 2: Métricas de Rendimiento del Modelo
- **Tarjetas KPI Globales:** Exactitud (Accuracy), AUC-ROC, F1-Score Macro, Recall y Precisión.
- **Curva ROC con AUC:** Gráfica interactiva frente a la línea de decisión aleatoria.
- **Matriz de Confusión Normalizada:** Mapa de calor visualizando verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.
- **Curvas de Pérdida y Precisión:** Comparativa entre entrenamiento y validación época a época.
- **Reporte de Clasificación:** Tabla detallada con precisión, recall y F1 por clase.

### Pestaña 3: Fases y Arquitectura del Sistema
- Desglose didáctico y académico de las 6 fases de la red convolucional.
- Glosario técnico completo: Convolución, Filtros/Kernels, Stride, Max Pooling, Separable Convolution y Funciones de Activación.
- Tabla detallada capa por capa con formas de salida, conteo de parámetros y propósito de diseño.
