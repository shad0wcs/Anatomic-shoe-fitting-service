"""
train_model.py — обучение CNN на синтетическом датасете
Transfer learning на MobileNetV2, три отдельных классификатора
"""

import os
import json
import numpy as np
import tensorflow as tf

from keras.applications import MobileNetV2
from keras import layers, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import image_dataset_from_directory

DATASET_DIR = "dataset"
MODELS_DIR  = "models"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20

os.makedirs(MODELS_DIR, exist_ok=True)

# Построение модели


def build_model(num_classes: int) -> Model:
    base = MobileNetV2(
        input_shape = (*IMG_SIZE, 3),
        include_top = False,
        weights     = "imagenet",
    )
    base.trainable = False

    x   = base.output
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base.input, outputs=out)



# Генераторы датасета (Keras 3 — image_dataset_from_directory)


def get_datasets(task_dir: str):
    train_ds = image_dataset_from_directory(
        task_dir,
        validation_split = 0.2,
        subset           = "training",
        seed             = 42,
        image_size       = IMG_SIZE,
        batch_size       = BATCH_SIZE,
        label_mode       = "categorical",
    )
    val_ds = image_dataset_from_directory(
        task_dir,
        validation_split = 0.2,
        subset           = "validation",
        seed             = 42,
        image_size       = IMG_SIZE,
        batch_size       = BATCH_SIZE,
        label_mode       = "categorical",
    )

    class_names = train_ds.class_names

    # Нормализация + аугментация для train
    augment = tf.keras.Sequential([
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15),
    ])
    normalize = tf.keras.Sequential([layers.Rescaling(1.0 / 255)])

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (normalize(x, training=False), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names



# Fine-tuning


def fine_tune(model: Model, train_ds, val_ds, task_name: str, epochs: int = 10):
    # В Keras 3 MobileNetV2 встроен как набор слоёв — ищем по имени
    mobilenet_layers = [l for l in model.layers if "mobilenet" in l.name.lower()]
    if mobilenet_layers:
        # Размораживаем верхние 30 слоёв MobileNetV2
        for layer in mobilenet_layers:
            layer.trainable = True
        # Замораживаем все слои кроме последних 30
        trainable_layers = [l for l in model.layers if l.trainable]
        for layer in trainable_layers[:-30]:
            layer.trainable = False
    else:
        # Размораживаем верхние 30 слоёв модели целиком
        for layer in model.layers[:-30]:
            layer.trainable = False
        for layer in model.layers[-30:]:
            layer.trainable = True
    print(f"Fine-tuning: разморожено {sum(1 for l in model.layers if l.trainable)} слоёв")

    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-5),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, f"{task_name}_best.keras"),
            save_best_only=True, monitor="val_accuracy"
        ),
    ]

    print(f"\n--- Fine-tuning: {task_name} ---")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model



# Обучение одного классификатора


def train_task(task_name: str):
    print(f"\n{'='*50}")
    print(f"Задача: {task_name}")

    task_dir = os.path.join(DATASET_DIR, task_name)
    train_ds, val_ds, class_names = get_datasets(task_dir)

    num_classes = len(class_names)
    print(f"Классы: {class_names}")

    model = build_model(num_classes)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-3),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, f"{task_name}_best.keras"),
            save_best_only=True, monitor="val_accuracy"
        ),
    ]

    print("\n--- Обучение головы ---")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Fine-tuning
    model = fine_tune(model, train_ds, val_ds, task_name)

    # Сохраняем в формате .keras (Keras 3)
    model_path = os.path.join(MODELS_DIR, f"{task_name}.keras")
    model.save(model_path)

    # Маппинг классов: {0: "название", 1: "название", ...}
    class_map = {str(i): name for i, name in enumerate(class_names)}
    with open(os.path.join(MODELS_DIR, f"{task_name}_classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n✓ {task_name}: val_accuracy = {acc:.3f}")
    return acc



# Запуск


if __name__ == "__main__":
    results = {}
    for task in ["arch", "pronation", "width"]:
        results[task] = train_task(task)

    print("\n" + "="*50)
    print("ИТОГИ ОБУЧЕНИЯ:")
    for task, acc in results.items():
        print(f"  {task:12s}: {acc:.1%}")

    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
