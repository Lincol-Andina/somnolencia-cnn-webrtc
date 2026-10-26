"""train.py
=============================================================================
Detección de Somnolencia en Conductores (Driver Drowsiness Detection - DDD)
Ingeniería de Deep Learning y Visión por Computadora
=============================================================================
Este script implementa:
1. Carga, división y preprocesamiento eficiente del dataset DDD.
2. Arquitectura CNN modular de 3 Ciclos Convolucionales:
   - Entrada: Reescalado y normalización [0, 1].
   - Bloque 1 (Temp-Conv): Conv2D con kernel adaptado a transiciones locales.
   - Normalización 1: BatchNormalization.
   - Bloque 2 (Spatial-Conv): Conv2D enfocado en rasgos faciales espaciales.
   - Activación y Pooling (Ciclo 2): ReLU + MaxPooling2D(2, 2).
   - Regularización 1: Dropout(0.3).
   - Bloque 3 (Separable-Conv): SeparableConv2D para inferencia ligera en tiempo real.
   - Activación y Pooling (Ciclo 3): ReLU + GlobalAveragePooling2D.
   - Clasificador MLP: Dense + Dropout + Dense(1, Sigmoid).
3. Entrenamiento con callbacks (EarlyStopping, ReduceLROnPlateau, Checkpoint).
4. Evaluación integral en conjunto de prueba independiente:
   - Curva ROC y cálculo de AUC.
   - Matriz de Confusión normalizada.
   - Curvas de pérdida (Loss) y precisión (Accuracy) por época.
   - Reporte de clasificación (Precision, Recall, F1-Score).
   - Exportación de modelo ('model_ddd.keras') y métricas ('metrics.json').
=============================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Asegurar codificación UTF-8 para salida en terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configuración de variables de entorno para suprimir advertencias innecesarias de TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica para generación de figuras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_seed(seed=42):
    """Fija las semillas aleatorias para garantizar reproducibilidad experimental."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_cnn_architecture(input_shape=(64, 64, 3)):
    """
    Construye la Red Neuronal Convolucional siguiendo estrictamente las fases y bloques
    especificados para la detección de somnolencia vehicular en tiempo real.
    
    Parámetros:
        input_shape (tuple): Dimensiones de entrada (H, W, C). Por defecto (64, 64, 3).
        
    Retorna:
        keras.Model: Modelo secuencial compilado o listo para entrenamiento.
    """
    model = keras.Sequential(
        [
            # 1. Entrada y Reescalado / Normalización de tensores a [0, 1]
            layers.Input(shape=input_shape, name="input_tensor"),
            layers.Rescaling(1.0 / 255.0, name="rescaling_normalization"),
            
            # 2. Bloque 1 (Temp-Conv): Captura patrones y micro-transiciones visuales
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="block1_temp_conv"
            ),
            
            # 3. Normalización 1: Estabilización de gradientes y aceleración de convergencia
            layers.BatchNormalization(name="batch_normalization_1"),
            
            # 4. Bloque 2 (Spatial-Conv): Detección de rasgos espaciales faciales (ojos, boca, postura)
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="block2_spatial_conv"
            ),
            
            # 5. Activación y Pooling (Ciclo 2): Reducción dimensional preservando rasgos dominantes
            layers.MaxPooling2D(pool_size=(2, 2), name="pooling_cycle2_maxpool"),
            
            # 6. Regularización 1: Dropout para mitigar sobreajuste
            layers.Dropout(0.3, name="dropout_regularization_1"),
            
            # 7. Bloque 3 (Separable-Conv): Convolución separable en profundidad (Depthwise + Pointwise)
            # Alta eficiencia computacional y bajo consumo de FLOPs para inferencia en tiempo real
            layers.SeparableConv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="block3_separable_conv"
            ),
            
            # 8. Activación y Pooling (Ciclo 3): Global Average Pooling para colapsar dimensiones espaciales
            layers.GlobalAveragePooling2D(name="pooling_cycle3_global_avg"),
            
            # 9. Clasificador (MLP): Capa densa intermedia con regularización y salida Sigmoid
            layers.Dense(64, activation="relu", name="classifier_dense_intermediate"),
            layers.Dropout(0.4, name="classifier_dropout"),
            layers.Dense(1, activation="sigmoid", name="classifier_output_sigmoid")
        ],
        name="CNN_Driver_Drowsiness_Detector"
    )
    return model


def load_datasets(data_dir, img_size=(64, 64), batch_size=64, max_samples=None, seed=42):
    """
    Carga el dataset DDD y realiza la partición en Train (70%), Validación (15%) y Prueba (15%).
    Mapeo de clases:
        0 -> 'Non Drowsy' (Alerta / Despierto)
        1 -> 'Drowsy'     (Somnoliento)
    """
    print(f"\n[+] Cargando imágenes desde: {data_dir}")
    class_names = ["Non Drowsy", "Drowsy"]

    # Carga partición de entrenamiento (70%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=class_names,
        validation_split=0.3,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    # Carga partición restante (30%) que se dividirá en Validación y Prueba
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=class_names,
        validation_split=0.3,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    total_val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_batches = total_val_test_batches // 2

    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    # Si se define un subconjunto de muestras para acelerar entrenamiento
    if max_samples and max_samples > 0:
        total_batches_requested = max_samples // batch_size
        train_b = max(4, int(total_batches_requested * 0.70))
        val_b = max(2, int(total_batches_requested * 0.15))
        test_b = max(2, int(total_batches_requested * 0.15))
        
        train_ds = train_ds.take(train_b)
        val_ds = val_ds.take(val_b)
        test_ds = test_ds.take(test_b)
        print(f"[i] Muestreo activo: {train_b} batches train, {val_b} batches val, {test_b} batches test.")

    # Optimización de pipeline de I/O mediante prefetching
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"[OK] Datasets preparados exitosamente.")
    return train_ds, val_ds, test_ds, class_names


def train_model(model, train_ds, val_ds, epochs=10, learning_rate=0.001, model_path="model_ddd.keras"):
    """
    Compila y entrena el modelo CNN utilizando optimizador Adam y pérdida de entropía cruzada binaria.
    """
    print("\n" + "=" * 60)
    print("           INICIO DE ENTRENAMIENTO DE LA CNN")
    print("=" * 60)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time
    print(f"\n[OK] Entrenamiento completado en {elapsed:.2f} segundos ({elapsed/60:.2f} minutos).")
    return history, elapsed


def evaluate_and_generate_metrics(model, test_ds, history, class_names, elapsed_time, output_dir="."):
    """
    Evalúa exhaustivamente el modelo en el conjunto de prueba independiente y genera:
    - Curva ROC y AUC
    - Matriz de Confusión normalizada
    - Curvas de entrenamiento (Loss y Accuracy)
    - Reporte de clasificación detallado
    - Archivo 'metrics.json'
    """
    print("\n" + "=" * 60)
    print("           EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print("=" * 60)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_true_list = []
    y_pred_probs_list = []

    print("[+] Generando inferencias en el conjunto de prueba...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy())
        y_pred_probs_list.extend(preds.flatten())

    y_true = np.array(y_true_list)
    y_pred_probs = np.array(y_pred_probs_list)
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)

    # 1. Métricas Globales
    test_acc = float(accuracy_score(y_true, y_pred_binary))
    test_prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    test_rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    test_f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

    # 2. Curva ROC y cálculo de AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = float(auc(fpr, tpr))

    # 3. Matriz de Confusión
    cm_raw = confusion_matrix(y_true, y_pred_binary)
    cm_norm = confusion_matrix(y_true, y_pred_binary, normalize="true")

    # 4. Reporte de Clasificación detallado
    clf_rep = classification_report(
        y_true,
        y_pred_binary,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    clf_rep_text = classification_report(
        y_true,
        y_pred_binary,
        target_names=class_names,
        zero_division=0
    )
    print("\n[Reporte de Clasificación]")
    print(clf_rep_text)
    print(f"Exactitud (Accuracy): {test_acc:.4f}")
    print(f"Área bajo la curva ROC (AUC): {roc_auc:.4f}")

    # =========================================================================
    # GENERACIÓN Y GUARDADO DE GRÁFICAS DE CALIDAD PROFESIONAL
    # =========================================================================

    # Estilo visual moderno y limpio
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # A) Curvas de Pérdida (Loss) y Precisión (Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    ax1.plot(epochs_range, history.history["accuracy"], "o-", color="#1f77b4", label="Entrenamiento (Train)", linewidth=2)
    ax1.plot(epochs_range, history.history["val_accuracy"], "s--", color="#ff7f0e", label="Validación (Val)", linewidth=2)
    ax1.set_title("Evolución de la Precisión (Accuracy)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Época", fontsize=11)
    ax1.set_ylabel("Exactitud", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs_range, history.history["loss"], "o-", color="#d62728", label="Entrenamiento (Train)", linewidth=2)
    ax2.plot(epochs_range, history.history["val_loss"], "s--", color="#9467bd", label="Validación (Val)", linewidth=2)
    ax2.set_title("Evolución de la Pérdida (Loss - Binary Crossentropy)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Época", fontsize=11)
    ax2.set_ylabel("Pérdida", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    training_curves_path = out_path / "training_curves.png"
    plt.savefig(training_curves_path, dpi=200)
    plt.close()
    print(f"[OK] Curvas de entrenamiento guardadas en: {training_curves_path}")

    # B) Curva ROC con AUC
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2ca02c", lw=2.5, label=f"Modelo CNN DDD (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Clasificador Aleatorio (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", fontsize=11, fontweight="bold")
    plt.title("Curva ROC (Receiver Operating Characteristic)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    roc_curve_path = out_path / "roc_curve.png"
    plt.savefig(roc_curve_path, dpi=200)
    plt.close()
    print(f"[OK] Curva ROC guardada en: {roc_curve_path}")

    # C) Matriz de Confusión Normalizada
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Matriz de Confusión Normalizada", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicción del Modelo", fontsize=11, fontweight="bold")
    plt.ylabel("Etiqueta Real (Ground Truth)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = out_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[OK] Matriz de Confusión guardada en: {cm_path}")

    # =========================================================================
    # ESTRUCTURACIÓN Y ALMACENAMIENTO DE METRICS.JSON
    # =========================================================================

    # Submuestreo suave de coordenadas ROC para optimizar almacenamiento JSON y renderizado
    step = max(1, len(fpr) // 100)
    fpr_sampled = fpr[::step].tolist()
    tpr_sampled = tpr[::step].tolist()
    if fpr[-1] != fpr_sampled[-1]:
        fpr_sampled.append(float(fpr[-1]))
        tpr_sampled.append(float(tpr[-1]))

    metrics_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(y_true),
        "training_time_seconds": round(elapsed_time, 2),
        "overall_metrics": {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1_score": round(test_f1, 4),
            "roc_auc": round(roc_auc, 4)
        },
        "confusion_matrix": {
            "classes": class_names,
            "raw": cm_raw.tolist(),
            "normalized": cm_norm.tolist()
        },
        "classification_report": clf_rep,
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr_sampled],
            "tpr": [round(float(x), 4) for x in tpr_sampled],
            "auc": round(roc_auc, 4)
        },
        "history": {
            "epochs": list(range(1, len(history.history["loss"]) + 1)),
            "train_loss": [round(float(x), 4) for x in history.history["loss"]],
            "val_loss": [round(float(x), 4) for x in history.history["val_loss"]],
            "train_accuracy": [round(float(x), 4) for x in history.history["accuracy"]],
            "val_accuracy": [round(float(x), 4) for x in history.history["val_accuracy"]]
        },
        "architecture_summary": {
            "name": model.name,
            "total_params": int(model.count_params()),
            "input_shape": list(model.input_shape[1:]),
            "layers_count": len(model.layers)
        }
    }

    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False)
    print(f"[OK] Métricas serializadas en: {metrics_file}")

    return metrics_payload


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento y Evaluación del Detector de Somnolencia DDD (Sprint 2)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Driver Drowsiness Dataset (DDD)",
        help="Ruta relativa o absoluta al directorio del dataset"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Tamaño de entrada espacial de las imágenes (ej. 64)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Tamaño del lote (batch size) para entrenamiento"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Número máximo de épocas de entrenamiento (EarlyStopping puede detener antes)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Número máximo de muestras (0 = dataset completo; use 9600 para una prueba rápida)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje del optimizador Adam"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_ddd.keras",
        help="Nombre/ruta de guardado del modelo Keras entrenado"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directorio donde se exportarán las métricas y gráficas"
    )

    args = parser.parse_args()
    set_seed(42)

    # 1. Cargar y particionar datasets
    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=42
    )

    # 2. Construir arquitectura de la red (3 Ciclos Convolucionales)
    model = build_cnn_architecture(input_shape=(args.img_size, args.img_size, 3))

    # 3. Entrenar red neuronal
    history, elapsed_time = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_path=args.model_path
    )

    # Asegurar que se guarde el modelo final en formato Keras
    model.save(args.model_path)
    print(f"\n[OK] Modelo guardado exitosamente en: {args.model_path}")

    # 4. Evaluación rigurosa y generación de artefactos
    evaluate_and_generate_metrics(
        model=model,
        test_ds=test_ds,
        history=history,
        class_names=class_names,
        elapsed_time=elapsed_time,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 60)
    print("   ¡PROCESO DE ENTRENAMIENTO Y EVALUACIÓN FINALIZADO CON ÉXITO!")
    print("=" * 60)


if __name__ == "__main__":
    main()
