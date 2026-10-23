# 🚗 GuardDrive AI - Detección de Somnolencia en Conductores en Tiempo Real

Sistema integral en Python para el monitoreo y detección de fatiga/somnolencia vehicular en tiempo real mediante Deep Learning y Visión por Computadora, desplegado en una aplicación web interactiva desarrollada con **Streamlit**.

---

## 📁 Estructura del Proyecto

```text
sprint 2/
│
├── Driver Drowsiness Dataset (DDD)/   # Dataset con 41,793 imágenes (227x227)
│   ├── Drowsy/                        # 22,348 imágenes (Clase 1: Somnoliento)
│   └── Non Drowsy/                    # 19,445 imágenes (Clase 0: Alerta / Despierto)
│
├── train.py                           # Pipeline de carga, entrenamiento, evaluación y exportación
├── app.py                             # Aplicación web interactiva Streamlit (Inferencia en vivo)
├── requirements.txt                   # Dependencias oficiales del entorno
├── README.md                          # Documentación técnica del proyecto
│
├── model_ddd.keras                    # Modelo entrenado Keras de 3 Ciclos Convolucionales
├── metrics.json                       # Métricas cuantitativas (ROC, AUC, Matriz de Confusión, F1)
├── roc_curve.png                      # Gráfica de la Curva ROC con AUC
├── confusion_matrix.png               # Gráfica de la Matriz de Confusión normalizada
├── training_curves.png                # Gráfica de curvas de Loss y Accuracy por época
└── alarm.wav                          # Archivo de audio para alerta acústica
```

---

## 🧠 Arquitectura de la Red Neuronal Convolucional (3 Ciclos Convolucionales)

La arquitectura de la red neuronal convolucional se implementa en la función `build_cnn_architecture()` del archivo [`train.py`](file:///d:/IA/sprint%202/train.py#L74-L138) (líneas 74 a 138). A continuación se detalla la ubicación exacta por **número de línea de código** para cada fase, bloque y ciclo convolucional:

| Etapa / Bloque | Capa Keras / Operación | Archivo | Líneas de Código | Propósito Técnico |
| :--- | :--- | :---: | :---: | :--- |
| **Definición Función** | `def build_cnn_architecture()` | `train.py` | [Líneas 74–85](file:///d:/IA/sprint%202/train.py#L74-L85) | Encapsulación modular de la CNN |
| **1. Entrada y Reescalado** | `layers.Input(shape=(64, 64, 3))` + `Rescaling(1./255)` | `train.py` | [Líneas 87–89](file:///d:/IA/sprint%202/train.py#L87-L89) | Normalización continua de píxeles a $[0.0, 1.0]$ |
| **2. Bloque 1 (Temp-Conv)** | `layers.Conv2D(32, (3, 3), padding='same', ReLU)` | `train.py` | [Líneas 91–98](file:///d:/IA/sprint%202/train.py#L91-L98) | Extracción de micro-transiciones y gradientes locales |
| **3. Normalización 1** | `layers.BatchNormalization()` | `train.py` | [Líneas 100–101](file:///d:/IA/sprint%202/train.py#L100-L101) | Estabilización del flujo de gradientes (Internal Covariate Shift) |
| **4. Bloque 2 (Spatial-Conv)** | `layers.Conv2D(64, (3, 3), padding='same', ReLU)` | `train.py` | [Líneas 103–110](file:///d:/IA/sprint%202/train.py#L103-L110) | Detección de rasgos espaciales faciales (ojos, boca, postura) |
| **5. Activación y Pooling (Ciclo 2)** | `layers.MaxPooling2D(pool_size=(2, 2))` | `train.py` | [Líneas 112–113](file:///d:/IA/sprint%202/train.py#L112-L113) | Reducción dimensional espacial ($64 \times 64 \to 32 \times 32$) |
| **6. Regularización 1** | `layers.Dropout(0.30)` | `train.py` | [Líneas 115–116](file:///d:/IA/sprint%202/train.py#L115-L116) | Prevención de co-adaptación de neuronas y sobreajuste |
| **7. Bloque 3 (Separable-Conv)** | `layers.SeparableConv2D(128, (3, 3), padding='same', ReLU)` | `train.py` | [Líneas 118–126](file:///d:/IA/sprint%202/train.py#L118-L126) | Convolución Depthwise + Pointwise (~88% menos FLOPs) |
| **8. Activación y Pooling (Ciclo 3)** | `layers.GlobalAveragePooling2D()` | `train.py` | [Líneas 128–129](file:///d:/IA/sprint%202/train.py#L128-L129) | Colapso de mapas espaciales a vector denso de 128D |
| **9. Clasificador (MLP) y Sigmoide** | `Dense(64, ReLU)` + `Dropout(0.40)` + `Dense(1, Sigmoid)` | `train.py` | [Líneas 131–134](file:///d:/IA/sprint%202/train.py#L131-L134) | Mapeo probabilístico final: $P(\text{Somnoliento} \mid X) \in [0, 1]$ |

### Detalle Específico por Línea en [`train.py`](file:///d:/IA/sprint%202/train.py):

1. **Entrada y Reescalado:** [Líneas 87–89](file:///d:/IA/sprint%202/train.py#L87-L89)
   - Línea 88: `layers.Input(shape=input_shape, name="input_tensor")`
   - Línea 89: `layers.Rescaling(1.0 / 255.0, name="rescaling_normalization")`
2. **Bloque 1 (Temp-Conv):** [Líneas 91–98](file:///d:/IA/sprint%202/train.py#L91-L98)
   - Líneas 92–98: `layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", name="block1_temp_conv")`
3. **Normalización 1:** [Líneas 100–101](file:///d:/IA/sprint%202/train.py#L100-L101)
   - Línea 101: `layers.BatchNormalization(name="batch_normalization_1")`
4. **Bloque 2 (Spatial-Conv):** [Líneas 103–110](file:///d:/IA/sprint%202/train.py#L103-L110)
   - Líneas 104–110: `layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="block2_spatial_conv")`
5. **Activación y Pooling (Ciclo 2):** [Líneas 112–113](file:///d:/IA/sprint%202/train.py#L112-L113)
   - Línea 113: `layers.MaxPooling2D(pool_size=(2, 2), name="pooling_cycle2_maxpool")`
6. **Regularización 1:** [Líneas 115–116](file:///d:/IA/sprint%202/train.py#L115-L116)
   - Línea 116: `layers.Dropout(0.3, name="dropout_regularization_1")`
7. **Bloque 3 (Separable-Conv):** [Líneas 118–126](file:///d:/IA/sprint%202/train.py#L118-L126)
   - Líneas 120–126: `layers.SeparableConv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", name="block3_separable_conv")`
8. **Activación y Pooling (Ciclo 3):** [Líneas 128–129](file:///d:/IA/sprint%202/train.py#L128-L129)
   - Línea 129: `layers.GlobalAveragePooling2D(name="pooling_cycle3_global_avg")`
9. **Clasificador (MLP) y Salida Sigmoide:** [Líneas 131–134](file:///d:/IA/sprint%202/train.py#L131-L134)
   - Línea 132: `layers.Dense(64, activation="relu", name="classifier_dense_intermediate")`
   - Línea 133: `layers.Dropout(0.4, name="classifier_dropout")`
   - Línea 134: `layers.Dense(1, activation="sigmoid", name="classifier_output_sigmoid")`

> **Nota de Implementación en [`app.py`](file:///d:/IA/sprint%202/app.py):**
> - La carga en caché del modelo compilado se realiza en la función `load_trained_model()` ([Líneas 155–168](file:///d:/IA/sprint%202/app.py#L155-L168)).
> - El preprocesamiento del tensor en vivo se ejecuta en `preprocess_frame()` ([Líneas 180–191](file:///d:/IA/sprint%202/app.py#L180-L191)).
> - La inferencia en tiempo real frame a frame se ejecuta en la [Línea 285](file:///d:/IA/sprint%202/app.py#L285): `prob_drowsy = float(model(tensor, training=False).numpy()[0][0])`.
> - La tabla y desglose académico didáctico se renderiza dinámicamente en la Pestaña 3 ([Líneas 840–860](file:///d:/IA/sprint%202/app.py#L840-L860)).

**Total de parámetros:** ~36,737 (~143.5 KB).

---

## 🚀 Instalación y Ejecución

### 1. Instalación de Dependencias
Abre una terminal en la carpeta `sprint 2/` e instala los paquetes necesarios:

```bash
pip install -r requirements.txt
```

### 2. Entrenamiento y Evaluación del Modelo (`train.py`)
Para reentrenar el modelo o generar nuevas métricas:

```bash
# Entrenamiento ágil recomendado (subconjunto balanceado para CPU):
python train.py --epochs 8 --batch_size 64 --max_samples 9600

# O entrenamiento con el dataset completo:
python train.py --epochs 10 --batch_size 64 --max_samples 0
```

Durante el entrenamiento se generarán y guardarán automáticamente:
- `model_ddd.keras` (Pesos y arquitectura serializada)
- `metrics.json` (Métricas cuantitativas completas)
- `roc_curve.png`, `confusion_matrix.png` y `training_curves.png`

### 3. Lanzamiento de la Aplicación Web (`app.py`)
Inicia el servidor local de Streamlit:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador web (por defecto `http://localhost:8501`).

---

## 🖥️ Módulos de la Aplicación Web (Streamlit)

### Pestaña 1: Detección en Tiempo Real (MVP)
- **Flujo de Video con OpenCV:** Captura frame a frame con `cv2.VideoCapture`.
- **HUD Integrado:** Superpone telemetría en vivo, FPS, estado ("Alerta / Despierto" o "¡SOMNOLENCIA DETECTADA!") y barra de probabilidad.
- **Alarma Sonora Acústica:** Si la somnolencia se sostiene durante $N$ cuadros consecutivos (configurable con un slider), se dispara una alarma audible en la laptop mediante un hilo en segundo plano (`winsound.Beep` / `pygame.mixer`).
- **Modos de Evaluación Alternativos:** Soporta carga de imágenes locales (`.jpg`, `.png`) y un evaluador con muestras aleatorias directas del dataset DDD.

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
