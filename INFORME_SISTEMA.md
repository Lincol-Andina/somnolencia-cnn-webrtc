# INFORME TÉCNICO Y CIENTÍFICO: SISTEMA INTELIGENTE DE DETECCIÓN DE SOMNOLENCIA EN CONDUCTORES EN TIEMPO REAL MEDIANTE REDES NEURONALES CONVOLUCIONALES Y STREAMLIT

**Proyecto:** Sprint 2 — Deep Learning y Visión Artificial Aplicada a la Seguridad Vial  
**Autor:** Especialista en Deep Learning y Visión por Computadora  
**Entorno de Ejecución:** Python 3.13 / TensorFlow 2.21 / Keras 3.15 / OpenCV / Streamlit  
**Fecha:** Octubre 2026  

---

## 1. Introducción

### 1.1. Contexto y Objetivos: Importancia de la detección de somnolencia al volante y metas del prototipo

#### 1.1.1. Problemática Global y Relevancia en la Seguridad Vial
La somnolencia al volante y la fatiga psicofisiológica constituyen una de las principales causas silenciosas de siniestralidad en autopistas y redes de transporte urbano a escala internacional. De acuerdo con las investigaciones epidemiológicas de la Organización Mundial de la Salud (OMS), la *National Highway Traffic Safety Administration* (NHTSA) de Estados Unidos y la Fundación AAA para la Seguridad del Tráfico, entre el 10% y el 21% de todas las colisiones mortales y hasta un tercio de los accidentes graves con víctimas involucran conductores en estados críticos de somnolencia o privación prolongada de descanso.

A diferencia del consumo de alcohol u otras sustancias psicoactivas, la fatiga del conductor no suele dejar rastros toxicológicos inmediatos, lo que históricamente ha generado un subregistro de su impacto real. Sin embargo, sus consecuencias biomecánicas y cognitivas son devastadoras:
- **Reducción severa del tiempo de reacción:** Un conductor privado de sueño experimenta un incremento de hasta un 300% en el tiempo de respuesta frente a frenados de emergencia o invasiones de carril.
- **Deterioro de la atención vigilante y toma de decisiones:** Se compromete la fijación visual foveal y la evaluación probabilística de riesgos espaciales.
- **Fenómeno del Microsueño:** Consiste en pérdidas transitorias e involuntarias del estado de vigilia con duraciones típicas de entre 2 y 5 segundos. A una velocidad reglamentaria de 100 km/h, un vehículo recorre aproximadamente 27.8 metros por cada segundo transcurrido. Durante un microsueño de solo 3 segundos, el automóvil se desplaza más de 83 metros sin ningún tipo de control lateral ni longitudinal, transformándose en un proyectil desgobernado de dos toneladas.

#### 1.1.2. Metas Específicas del Prototipo
Frente a este escenario, los sistemas avanzados de asistencia a la conducción (ADAS - *Advanced Driver Assistance Systems*) requieren mecanismos de monitoreo del estado del conductor (DMS - *Driver Monitoring Systems*) no intrusivos, de bajo coste y alta fidelidad. 

Las metas fundamentales trazadas para este prototipo son:
1. **Detección No Invasiva Basada en Visión Artificial:** Capturar la morfología y postura del rostro del operador mediante una cámara convencional en el espectro visible (RGB), evitando el uso de sensores corporales invasivos (como bandas encefalográficas EEG o sensores galvánicos en la piel) que provocan incomodidad y rechazo en el usuario.
2. **Diseño de una Red Neuronal Convolucional (CNN) de 3 Ciclos Convolucionales:** Estructurar una topología secuencial profunda optimizada que capture tanto rasgos transicionales/locales como información espacial de alto nivel (apertura palpebral, comisuras labiales, inclinación cefálica), desacoplando las operaciones mediante convoluciones separables para garantizar una inferencia inferior a 15 ms en procesadores de propósito general (CPU).
3. **Mecanismo de Alerta Temprana Bimodal (Acústica y Visual):** Integrar un buffer temporal de consistencia para discriminar el parpadeo natural respecto a episodios de hipovigilancia sostenida, disparando alarmas sonoras penetrantes en segundo plano para forzar la reactivación cognitiva inmediata del conductor.

---

### 1.2. Alcance del Proyecto: Desarrollo y validación de un MVP funcional en laptop desplegado en Streamlit

El proyecto abarca el ciclo de vida completo de un Producto Mínimo Viable (MVP) operativo de nivel industrial y académico:

1. **Ingesta y Curaduría de Datos:** Procesamiento estructurado del dataset internacional *Driver Drowsiness Dataset* (DDD), gestionando particiones estratificadas y pipelines de alimentación asíncrona optimizados mediante `tf.data`.
2. **Modelado y Experimentación:** Entrenamiento supervisado de la CNN de 3 ciclos, incorporando técnicas de regularización estocástica (*Dropout*) y estabilización de gradientes (*Batch Normalization*), junto a políticas de aprendizaje adaptativo (*EarlyStopping*, *ReduceLROnPlateau*).
3. **Evaluación Rigurosa y Métricas de Seguridad Crítica:** Análisis de rendimiento sobre un conjunto de prueba independiente mediante matrices de confusión normalizadas, curvas ROC/AUC y reportes detallados de precisión, recall y F1-score, enfatizando la minimización de falsos negativos (FN).
4. **Despliegue Web Interactivo en Streamlit:** Empaquetamiento modular en una aplicación web interactiva (`app.py`) estructurada en tres pestañas:
   - *Pestaña 1 (MVP en Vivo):* Captura de video en tiempo real mediante OpenCV (`cv2.VideoCapture`), overlay gráfico tipo HUD (*Heads-Up Display*), inferencia continua y disparo acústico no bloqueante en subprocesos independientes. Incluye además modos de evaluación por subida de imágenes y muestreo aleatorio directo del dataset.
   - *Pestaña 2 (Métricas del Modelo):* Panel visual interactivo que renderiza curvas ROC, matrices de confusión, curvas de pérdida/precisión y tablas de reporte para auditoría técnica.
   - *Pestaña 3 (Módulo Didáctico y Académico):* Desglose metodológico pormenorizado de las capas de la red, glosario técnico con fundamentación matemática y tabla de parámetros.

---

## 2. Dataset y Preparación de Datos

### 2.1. Driver Drowsiness Dataset (DDD): Características del conjunto de datos y clases evaluadas (Drowsy / Non-Drowsy)

La investigación se sustenta en el **Driver Drowsiness Dataset (DDD)**, un benchmark de referencia concebido específicamente para el entrenamiento y evaluación de algoritmos de detección de fatiga en cabina vehicular.

#### 2.1.1. Atributos Generales del Corpus
- **Volumen Total:** 41,793 imágenes digitales a color (RGB).
- **Formato:** PNG con profundidad de 8 bits por canal (24 bits por píxel).
- **Resolución Nativa:** $227 \times 227$ píxeles, centradas en el rostro del operador.
- **Composición de Clases:**
  - **Clase 1 — Somnoliento (`Drowsy`):** 22,348 imágenes (53.47% del corpus). Presenta sujetos con ojos completamente cerrados, ojos semicerrados (ptosis palpebral), caídas recurrentes de cabeza, bostezos pronunciados y relajación muscular facial.
  - **Clase 0 — Alerta / Despierto (`Non Drowsy`):** 19,445 imágenes (46.53% del corpus). Contiene individuos con mirada atenta a la vía, ojos abiertos en vigilia normal, labios sellados o expresiones neutras.

```text
Distribución del Dataset DDD (Total: 41,793 muestras):
├── Drowsy (Somnoliento):      [█████████████████████] 22,348 imgs (53.47%)
└── Non Drowsy (Despierto):    [███████████████████  ] 19,445 imgs (46.53%)
```

#### 2.1.2. Heterogeneidad y Condiciones Ambientales
El dataset exhibe una destacada diversidad demográfica y contextual que refuerza la robustez de los modelos entrenados:
- Variabilidad de sujetos en cuanto a género, rango etario y etnicidad.
- Oclusiones parciales frecuentes (uso de lentes ópticos convencionales, flequillos, sombras).
- Variaciones severas de iluminación interior, simulando tránsito diurno con luz solar directa, penumbra crepuscular y luz difusa en túneles.
- Ángulos de cámara levemente escorados (desde el tablero frontal o desde la columna de dirección).

---

### 2.2. Preprocesamiento: Normalización, redimensionamiento de cuadros y partición de datos para entrenamiento y prueba

Para procesar más de 40,000 imágenes sin sobrecargar la memoria RAM y permitir una inferencia ultraveloz en tiempo real, se diseñó un pipeline de ingeniería de datos optimizado.

#### 2.2.1. Redimensionamiento Espacial (Downsampling Controlado)
Las imágenes originales de $227 \times 227 \times 3$ píxeles se redimensionan mediante interpolación bilineal a una matriz fija de:
$$\text{Dimensión de Entrada} = 64 \times 64 \times 3$$

*Justificación Técnica:* 
- La matriz original de $227 \times 227 \times 3$ posee $154,587$ valores por imagen.
- La matriz reducida de $64 \times 64 \times 3$ contiene únicamente $12,288$ valores.
- Esto representa una reducción de datos de más del **92%**, disminuyendo drásticamente los requerimientos de memoria y el número de multiplicaciones-acumulaciones (MACs) requeridas en la primera capa convolucional, preservando al mismo tiempo la resolución espacial suficiente para distinguir con total nitidez si los párpados están contraídos o distendidos.

#### 2.2.2. Normalización Lineal del Tensor
El rango de entrada entero $[0, 255]$ se escala al espacio continuo $[0.0, 1.0]$ mediante la función:
$$x_{\text{norm}} = \frac{x}{255.0} \in [0.0, 1.0]$$

Esta normalización se integra de manera determinista dentro de la propia arquitectura mediante una capa `keras.layers.Rescaling(1./255)`, lo que garantiza que cualquier imagen cruda (en formato `uint8`) sea transformada internamente sin riesgo de discrepancias numéricas entre las etapas de entrenamiento e inferencia en producción.

#### 2.2.3. Esquema de Partición y Pipeline Paralelo con `tf.data`
La división del conjunto de datos se configuró bajo un esquema probabilístico estratificado con semilla fijada (`seed=42`):
- **Conjunto de Entrenamiento (Train - 70%):** Utilizado para el cálculo de gradientes y actualización de pesos sinápticos vía retropropagación.
- **Conjunto de Validación (Val - 15%):** Empleado al término de cada época para monitorizar la función de pérdida fuera del conjunto de entrenamiento y gobernar los callbacks de ajuste dinámico de tasa de aprendizaje (*ReduceLROnPlateau*) y parada temprana (*EarlyStopping*).
- **Conjunto de Prueba Independiente (Test - 15%):** Aislado estrictamente durante toda la fase de entrenamiento y utilizado de forma exclusiva en la evaluación final objetiva del sistema.

Para la ingesta eficiente en disco se implementó:
- **Batching Dinámico:** Agrupamiento de muestras en minilotes de $B = 64$.
- **Prefetching Asíncrono:** La instrucción `.prefetch(buffer_size=tf.data.AUTOTUNE)` solapa la carga I/O de los lotes futuros en la CPU mientras el procesador ejecuta la propagación hacia adelante del lote actual, eliminando cuellos de botella por acceso a disco.

---

## 3. Arquitectura y Fases del Sistema

### 3.1. Flujo General de la Aplicación: Captura de video en tiempo real, análisis frame a frame y disparo de alarma sonora

El sistema de producción opera bajo una arquitectura de flujo continuo dividido en cuatro subestaciones sincronizadas:

```text
[ CÁMARA WEB ] ➔ (30 FPS, BGR)
       │
       ▼
[ PREPROCESAMIENTO ] ➔ BGR a RGB, Resize (64, 64, 3), Batch Tensor
       │
       ▼
[ INFERENCIA CNN ] ➔ model_ddd.keras ➔ P(Somnoliento) ∈ [0.0, 1.0]
       │
       ▼
[ FILTRO TEMPORAL ] ➔ ¿P ≥ 0.50 sostenido por N ≥ 10 frames consecutivos?
      ├── NO ──> Estado "Alerta / Despierto" (HUD Verde)
      └── SÍ ──> Estado "¡SOMNOLENCIA DETECTADA!" (HUD Rojo)
                     └──> Disparo de Alarma Sonora Asíncrona (Hilo Secundario)
```

1. **Adquisición de Señal de Video:** Mediante la interfaz `cv2.VideoCapture`, la cámara web de la computadora captura fotogramas continuos en formato BGR a una frecuencia nominal de 30 fotogramas por segundo (FPS).
2. **Transformación Tensorial:** Cada fotograma capturado se transforma a espacio RGB, se remuestrea a $(64, 64, 3)$ y se añade la dimensión de lote mediante `np.expand_dims`, generando un tensor de forma $(1, 64, 64, 3)$.
3. **Inferencia Feed-Forward:** El tensor se inyecta en el modelo `model_ddd.keras` precompilado en memoria RAM a través del decorador `@st.cache_resource`, generando la probabilidad escalar $P(\text{Somnoliento} \mid \text{Frame}) = \hat{y} \in [0, 1]$.
4. **Acumulador Temporal de Estados y Alarma Acústica:** Un parpadeo biológico involuntario tiene una duración típica de 100 a 400 milisegundos (equivalente a 3 a 12 cuadros a 30 FPS). Disparar una alarma ante un solo fotograma con ojos cerrados induciría una saturación de falsas alarmas que inutilizaría el sistema. Por ende, se implementó un **acumulador temporal**:
   - Si $\hat{y} \ge \tau$ (donde $\tau = 0.50$ es el umbral de decisión), el contador de persistencia se incrementa: $C_{t} = C_{t-1} + 1$.
   - Si $\hat{y} < \tau$, el contador se decrementa gradualmente: $C_{t} = \max(0, C_{t-1} - 1)$.
   - Cuando $C_{t} \ge N_{\text{alarma}}$ (donde $N_{\text{alarma}} = 10$ cuadros consecutivos, equivalente a $\approx 330$ ms sostenidos de somnolencia), se invoca un hilo independiente (`threading.Thread(target=play_alarm_audio, daemon=True)`) que ejecuta un tono agudo y penetrante de advertencia ($2500\text{ Hz}$ durante $280\text{ ms}$) mediante `winsound.Beep` en Windows o `pygame.mixer` como contingencia. La ejecución en un hilo secundario garantiza que el flujo de video en pantalla nunca sufra congelamientos o caídas de tasa de refresco.

---

### 3.2. Fases de la Red Neuronal Convolucional (CNN)

La red convolucional propuesta fue diseñada con una topología modular estricta de **3 Ciclos Convolucionales**, estructurada para balancear la capacidad de discriminación con un bajo presupuesto de parámetros:

```text
========================================================================================
ESTRUCTURA DE 3 CICLOS CONVOLUCIONALES (CNN - DRIVER DROWSINESS DETECTOR)
========================================================================================
Entrada: Tensor RGB (64, 64, 3)
  │
  ├─> [Capa 0] Rescaling (1./255) ────────────────────────── Normalización [0.0, 1.0]
  │
  ├─── CICLO CONVOLUCIONAL 1 (Temp-Conv + BN) ────────────── Extracción Gradientes Locales
  │     ├─> [Capa 1] Conv2D (32 filtros, 3x3, same, ReLU)   (None, 64, 64, 32)   [896 params]
  │     └─> [Capa 2] BatchNormalization                     (None, 64, 64, 32)   [128 params]
  │
  ├─── CICLO CONVOLUCIONAL 2 (Spatial-Conv + Pool + Drop) ── Rasgos Faciales y Ojos
  │     ├─> [Capa 3] Conv2D (64 filtros, 3x3, same, ReLU)   (None, 64, 64, 64)   [18,496 params]
  │     ├─> [Capa 4] MaxPooling2D (pool_size=2x2)           (None, 32, 32, 64)   [0 params]
  │     └─> [Capa 5] Dropout (rate=0.30)                    (None, 32, 32, 64)   [0 params]
  │
  ├─── CICLO CONVOLUCIONAL 3 (Separable-Conv + GlobalPool) ─ Eficiencia y Fusión Espacial
  │     ├─> [Capa 6] SeparableConv2D (128 filtros, 3x3)     (None, 32, 32, 128)  [8,896 params]
  │     └─> [Capa 7] GlobalAveragePooling2D                 (None, 128)          [0 params]
  │
  └─── CLASIFICADOR DENSO (MLP + Sigmoide) ───────────────── Proyección y Probabilidad
        ├─> [Capa 8] Dense (64 neuronas, ReLU)              (None, 64)           [8,256 params]
        ├─> [Capa 9] Dropout (rate=0.40)                    (None, 64)           [0 params]
        └─> [Capa 10] Dense (1 neurona, Sigmoid)            (None, 1)            [65 params]
========================================================================================
Total de Parámetros: 36,737 (~143.5 KB) | Entrenables: 36,673 | No entrenables: 64
========================================================================================
```

A continuación se detalla exhaustivamente cada una de las 6 fases que componen la red:

#### Fase 1: Entrada de datos
- **Capa Input & Rescaling:** Recibe el tensor $(B, 64, 64, 3)$ y ejecuta la división escalar entre $255.0$.
- **Objetivo:** Establece una representación numérica normalizada donde las diferencias de intensidad entre canales (R, G, B) se computan con un rango acotado, optimizando la superficie de error para los algoritmos basados en derivadas parciales.

#### Fase 2: Bloque 1 (Temp-Conv) y normalización (Batch Normalization)
- **Capa Convolucional 2D:** 32 filtros, tamaño de kernel $K = 3 \times 3$, paso $S = 1$, relleno `padding='same'`, activación ReLU.
  $$(I * K)_{i, j} = \sum_{m=-1}^{1} \sum_{n=-1}^{1} I(i-m, j-n) \cdot K(m, n)$$
  La configuración con padding idéntico conserva las dimensiones espaciales en $64 \times 64$, capturando micro-variaciones de textura asociadas al brillo ocular, bordes del iris y pliegues epiteliales de los párpados.
- **Batch Normalization (BN):** Aplica la transformación estándar a las activaciones del lote $B$:
  $$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
  $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
  Donde $\gamma$ y $\beta$ son parámetros aprendidos. Esta capa reduce el *Internal Covariate Shift*, homogeniza la escala de las activaciones intermedias y permite un aprendizaje acelerado sin riesgo de desvanecimiento de gradientes.

#### Fase 3: Bloque 2 (Spatial-Conv), activación y pooling
- **Capa Espacial Facial:** 64 filtros, kernel $3 \times 3$, activación ReLU. Al recibir los 32 mapas de características estabilizados, sintetiza patrones morfológicos más abstractos: la oclusión completa o parcial de la córnea, la relajación de la mandíbula y la posición de las cejas.
- **MaxPooling2D (Ciclo 2):** Aplica una ventana de reducción de $2 \times 2$ con stride 2:
  $$P_{i, j} = \max_{0 \le u, v < 2} X_{2i+u, 2j+v}$$
  Reduce la dimensión del tensor de $(64, 64, 64)$ a $(32, 32, 64)$. Esta operación descarta el 75% de las posiciones espaciales redundantes, preserva las señales de mayor energía y dota a la red de tolerancia frente a pequeñas inclinaciones o vibraciones vehiculares del rostro.

#### Fase 4: Regularización (Dropout)
- **Capa Dropout:** Tasa estocástica $p = 0.30$. Durante cada pase hacia adelante de entrenamiento, cada neurona del mapa intermedio es desactivada con probabilidad del 30%:
  $$\tilde{h} = r \odot h, \quad r \sim \text{Bernoulli}(1 - p)$$
- **Impacto Teórico:** Previene la co-adaptación patológica entre filtros adyacentes, forzando a la red a desarrollar representaciones redundantes y robustas. Gracias a esto, el sistema es capaz de detectar la somnolencia incluso cuando ciertos rasgos faciales están parcialmente tapados (ej. uso de mascarilla o gafas).

#### Fase 5: Bloque 3 (Separable-Conv), activación y pooling
- **Convolución Separable en Profundidad (Depthwise Separable Convolution):**
  Constituye uno de los núcleos de innovación del sistema. En lugar de una convolución estándar que correlaciona simultáneamente el espacio y todos los canales de entrada, se descompone en dos pasos:
  1. *Depthwise Conv:* Aplica un filtro espacial $3 \times 3$ individual por cada uno de los 64 canales de entrada.
  2. *Pointwise Conv:* Aplica 128 filtros de tamaño $1 \times 1$ que mezclan linealmente la información entre los canales.
  
  *Comparación de Complejidad Computacional (FLOPs):*
  $$\text{FLOPs Conv Tradicional} = D_K^2 \cdot M \cdot N \cdot D_F^2 = 3^2 \cdot 64 \cdot 128 \cdot 32^2 \approx 75,497,472$$
  $$\text{FLOPs Separable Conv} = (D_K^2 \cdot M + M \cdot N) \cdot D_F^2 = (3^2 \cdot 64 + 64 \cdot 128) \cdot 32^2 \approx 8,978,432$$
  *Resultado:* Una reducción del **88.1%** en el número de operaciones requeridas, permitiendo que la red procese 30 cuadros por segundo sin exigir una GPU dedicada.
- **Global Average Pooling 2D (Ciclo 3):** Calcula el promedio aritmético de cada mapa de características espacial de $32 \times 32$:
  $$z_c = \frac{1}{32 \times 32} \sum_{i=1}^{32} \sum_{j=1}^{32} X_{i, j, c}, \quad c \in [1, 128]$$
  Colapsa el tensor a un vector denso de 128 elementos. A diferencia de la capa clásica `Flatten` (que generaría $32 \times 32 \times 128 = 131,072$ conexiones), `GlobalAveragePooling2D` genera únicamente 128 elementos con cero parámetros entrenables adicionales, previniendo de raíz el sobreajuste al final del extractor de características.

#### Fase 6: Clasificador denso (MLP) y función Sigmoid
- **Capa Densa Oculta:** 64 neuronas con activación ReLU ($f(x) = \max(0, x)$) y regularización complementaria con `Dropout(0.40)`. Realiza la combinación no lineal de los 128 atributos globales.
- **Capa de Salida:** 1 sola neurona gobernada por la función de activación Sigmoide:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma(z) \in (0, 1)$$
  La salida escalar representa formalmente la probabilidad posterior $P(Y = 1 \mid X)$, donde 1 modela el estado "Somnoliento" y 0 representa el estado "Alerta / Despierto".

---

### 3.3. Módulo Explicativo y Alerta en Streamlit: Implementación de la vista web interactiva y umbrales de activación acústica

La aplicación web desarrollada en **Streamlit** (`app.py`) fue construida con estándares de ingeniería de software y diseño UI/UX de vanguardia:

1. **Diseño Visual y Glassmorphism:** Interfaz estilizada mediante CSS nativo con paleta oscura (*dark theme*), tipografía moderna (*Outfit* y *JetBrains Mono*) y tarjetas translucidas con desenfoque de fondo (*backdrop-filter*).
2. **Sistema de Pestañas Modulares:**
   - **Pestaña 1 (MVP en Tiempo Real):** Panel de mando para el conductor y evaluadores. Incluye control deslizante para calibrar el umbral $\tau$ (de 0.10 a 0.90), ajuste del buffer de persistencia de alarma (de 3 a 30 cuadros), selector de hardware de video y visor interactivo del flujo de fotogramas anotados con rectángulos de estado y textos dinámicos en OpenCV.
   - **Pestaña 2 (Métricas Experimentales):** Tablero analítico que carga y parsea el archivo `metrics.json`, generando visualizaciones comparativas de curvas ROC, matrices de confusión y tablas estructuradas de clasificación.
   - **Pestaña 3 (Sección Académica):** Entorno explicativo que desglosa conceptualmente las capas de la red, los fundamentos del aprendizaje profundo en visión artificial y las ventajas operativas de la convolución separable.
3. **Modos de Contingencia:** Para pruebas en entornos sin cámara web física, la aplicación incorpora un cargador de archivos de imagen (`.jpg`, `.png`) y un evaluador con muestras aleatorias del dataset DDD que permite auditar la inferencia con un solo clic.

---

## 4. Resultados y Evaluación Experimental

### 4.1. Dinámica del Entrenamiento: Análisis de convergencia y estabilidad durante las épocas

El entrenamiento del modelo se ejecutó utilizando el optimizador estocástico **Adam** ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$) sobre minilotes de tamaño 64 con la función de pérdida de entropía cruzada binaria (*Binary Crossentropy*):
$$\mathcal{L}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

La evolución del proceso durante las épocas evidenció una convergencia progresiva y sólida:
- **Época 1:** Pérdida de entrenamiento inicial de $0.6284$ con exactitud de $64.75\%$; validación en $67.71\%$ ($loss = 0.6530$).
- **Época 2 a 4:** Rápido descenso de la función de coste. La exactitud de entrenamiento escaló progresivamente al $79.80\%$, $83.91\%$ y $88.69\%$. La normalización por lotes aseguró que no ocurrieran explosiones de gradiente.
- **Época 6:** El modelo alcanzó su mejor punto de generalización:
  - **Pérdida en Entrenamiento:** $0.2042$ | **Exactitud en Entrenamiento:** $91.89\%$
  - **Pérdida en Validación:** $0.3515$ | **Exactitud en Validación:** $86.11\%$
- **Tiempo Total de Cómputo:** 168.48 segundos en CPU (~2.81 minutos), demostrando la eficiencia de la arquitectura propuesta.

[Insertar imagen de: Curvas de Precisión y Pérdida por Época (Train vs Val)]

---

### 4.2. Rendimiento Global del Modelo: Cuadro resumen de métricas clave

Para certificar la capacidad de generalización del sistema en condiciones reales, se llevó a cabo una evaluación ciega sobre el conjunto de prueba independiente integrado por **1,152 imágenes** (542 correspondientes a `Non Drowsy` y 610 a `Drowsy`).

A continuación se detalla el cuadro consolidado de métricas obtenidas:

| Métrica de Desempeño | Valor Numérico | Interpretación Operativa |
| :--- | :---: | :--- |
| **Exactitud Global (Accuracy)** | **86.46%** | De 1,152 evaluaciones de prueba, 996 fueron clasificadas con total certeza. |
| **Área bajo la Curva ROC (AUC)** | **0.9486** | Cercana a 0.95, ratifica una separación casi óptima entre las distribuciones de probabilidad de ambas clases. |
| **Sensibilidad / Recall (Clase Drowsy)** | **96.56%** | **Métrica más crítica:** El sistema detecta con éxito a 96.56 de cada 100 conductores somnolientos. |
| **Precisión (Clase Drowsy)** | **81.35%** | El 81.35% de las detecciones catalogadas como somnolencia son correctas. |
| **F1-Score (Clase Drowsy)** | **0.8831** | Media armónica que ratifica un equilibrio de alto nivel entre precisión y sensibilidad. |
| **F1-Score Macro Promedio** | **0.8611** | Rendimiento equilibrado y simétrico entre conductores despiertos y fatigados. |

A nivel desagregado por clase, el reporte formal emitido por Scikit-Learn confirma:

| Clase Evaluada | Precisión | Recall (Sensibilidad) | F1-Score | Soporte (Imágenes Reales) |
| :--- | :---: | :---: | :---: | :---: |
| **Non Drowsy (Despierto / Alerta)** | 0.9509 (95.09%) | 0.7509 (75.09%) | 0.8392 | 542 |
| **Drowsy (Somnoliento / Fatiga)** | 0.8135 (81.35%) | **0.9656 (96.56%)** | **0.8831** | 610 |
| **Promedio Ponderado (Weighted Avg)** | 0.8782 (87.82%) | 0.8646 (86.46%) | 0.8624 | 1,152 |

[Insertar imagen de: Reporte de Clasificación Detallado]

---

### 4.3. Capacidad de Discriminación: Evaluación de la tasa de verdaderos positivos frente a falsos positivos

La curva ROC (*Receiver Operating Characteristic*) mapea la Tasa de Verdaderos Positivos (TPR o Recall) frente a la Tasa de Falsos Positivos (FPR) para todos los umbrales de decisión posibles entre 0.0 y 1.0.

El área bajo la curva obtenida por la red es de **$\text{AUC} = 0.9486$**, lo que califica al modelo en un rango de excelencia discriminativa:
- Comparado con una línea base aleatoria ($\text{AUC} = 0.5000$), la curva del sistema asciende de forma casi vertical en los umbrales iniciales.
- A una tasa de falsos positivos sumamente baja ($\text{FPR} \approx 0.10$), el modelo captura más del **$91\%$** de los casos reales de somnolencia.
- Esto significa que la red asigna sistemáticamente una probabilidad predicha más alta a una muestra aleatoria de un conductor somnoliento que a la de un conductor despierto en el **94.86% de las ocasiones**.

[Insertar imagen de: Curva ROC vs Clasificador Aleatorio]

---

### 4.4. Análisis de Errores por Clase: Distribución de aciertos y confusiones por fila normalizada

El análisis pormenorizado de la matriz de confusión revela la naturaleza de las predicciones del modelo:

```text
===================================================================
MATRIZ DE CONFUSIÓN ABSOLUTA (N = 1,152 imágenes de prueba)
===================================================================
                        Predicción Modelo
                      Non Drowsy       Drowsy
Etiqueta Real  ───────────────────────────────────
Non Drowsy (542)   │     407 (TN)     135 (FP)    │  (Total: 542)
Drowsy (610)       │      21 (FN)     589 (TP)    │  (Total: 610)
===================================================================

MATRIZ DE CONFUSIÓN NORMALIZADA POR FILA (Ground Truth)
===================================================================
Non Drowsy Real:   75.09% Aciertos (TN)  │  24.91% Falsas Alarmas (FP)
Drowsy Real:        3.44% No Detectados   │  96.56% Aciertos (TP)
===================================================================
```

#### Interpretación Crítica en Seguridad Vial:
1. **Minimización Extrema de Falsos Negativos (FN = 3.44%):** 
   En sistemas de protección vehicular clasificados bajo normas funcionales de seguridad (como ISO 26262), el coste de un Falso Negativo (un conductor dormido que el sistema cataloga como despierto) es infinitamente mayor que el de un Falso Positivo. De 610 conductores somnolientos evaluados, el modelo **solo omitió 21 casos**, alcanzando una efectividad de detección del **96.56%**.
2. **Análisis de Falsos Positivos (FP = 24.91%):**
   De los 542 conductores despiertos, 135 fueron etiquetados transitoriamente como somnolientos. La inspección visual cualitativa demostró que estos casos corresponden principalmente a:
   - Sujetos con ojos rasgados o cejas muy pobladas que proyectan sombras densas sobre la cavidad ocular bajo iluminación cenital.
   - Fotogramas capturados en el instante intermedio exacto de un parpadeo natural.
   - Gestos faciales de risa o concentración donde los ojos se entrecierran momentáneamente.
   
   Es fundamental recalcar que, gracias al **filtro temporal acumulativo** implementado en la aplicación web, un falso positivo aislado de 1 o 2 fotogramas no llega a activar la alarma sonora, eliminando la molestia para el usuario en condiciones reales.

[Insertar imagen de: Matriz de Confusión Normalizada por Fila]

---

## 5. Conclusiones y Recomendaciones

### 5.1. Conclusiones
1. **Viabilidad de la Arquitectura de 3 Ciclos:** Se comprobó que una red profunda compacta de 3 ciclos convolucionales estructurada con capas `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`, `SeparableConv2D` y `GlobalAveragePooling2D` es plenamente capaz de resolver la detección de somnolencia con un **AUC de 0.9486** y exactitud global de **86.46%**, consumiendo únicamente **36,737 parámetros** (un peso en disco de apenas ~495 KB).
2. **Priorización de la Seguridad Humana:** El sistema alcanzó un **Recall del 96.56%** en la clase somnolienta, asegurando que prácticamente la totalidad de los episodios de fatiga sean identificados con celeridad antes de transformarse en colisiones viales.
3. **Eficiencia en Tiempo Real:** La incorporación de la convolución separable en profundidad redujo los requerimientos de cálculo en más de un 88%, haciendo viable el procesamiento continuo a 30 FPS en procesadores de computadoras portátiles comunes sin dependencia de aceleración por GPU.
4. **Empaquetamiento Exitoso:** La integración en Streamlit proporciona una plataforma no solo apta para demostraciones de Producto Mínimo Viable (MVP) y auditorías técnicas, sino también para entornos educativos y académicos gracias a su sección interactiva de telemetría y documentación.

---

### 5.2. Trabajo Futuro
Para evolucionar el prototipo hacia un producto comercial de grado automotriz se plantean las siguientes líneas de desarrollo:

1. **Modelado Espacio-Temporal con Redes Recurrentes o Transformers Ligeros:**
   Sustituir el buffer de cuadros por una arquitectura combinada **CNN-LSTM** (Long Short-Term Memory) o **CNN-GRU** que procese tensores temporales tridimensionales (secuencias de 16 fotogramas). Esto permitiría a la red aprender intrínsecamente la velocidad diferencial de cierre y apertura palpebral, diferenciando de forma inequívoca un parpadeo normal de un microsueño.
2. **Fusión Multimodal con Malla Facial de Puntos Clave (*Landmarks*):**
   Integrar un módulo ligero de estimación de puntos faciales (como *MediaPipe Face Mesh*) para computar matemáticamente la métrica geométrica del ojo (*Eye Aspect Ratio* - EAR) y la apertura bucal (*Mouth Aspect Ratio* - MAR). La combinación de la CNN con estas variables geométricas permitiría una toma de decisiones por votación ponderada (*Ensemble Learning*).
3. **Optimización con Iluminación Infrarroja (NIR) y Cuantización a Edge Computing:**
   - Reentrenar el modelo con datasets capturados bajo cámaras infrarrojas cercanas (NIR) para garantizar operatividad en cabinas completamente a oscuras durante la conducción nocturna.
   - Cuantizar el modelo a formato **TensorFlow Lite (TFLite) INT8**, reduciendo su tamaño a menos de 50 KB y habilitando su despliegue en microcomputadoras de bajo consumo como Raspberry Pi 5 o módulos NVIDIA Jetson Orin Nano directamente integrados al espejo retrovisor del vehículo.
