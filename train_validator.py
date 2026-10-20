"""
train_validator.py — обучение бинарного классификатора
стопа / не_стопа на MobileNetV2
"""

import os
import json
import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import image_dataset_from_directory

DATASET_DIR = "dataset/validate"
MODELS_DIR  = "models"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20

os.makedirs(MODELS_DIR, exist_ok=True)


def get_datasets():
    train_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split = 0.2,
        subset           = "training",
        seed             = 42,
        image_size       = IMG_SIZE,
        batch_size       = BATCH_SIZE,
        label_mode       = "binary",
    )
    val_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split = 0.2,
        subset           = "validation",
        seed             = 42,
        image_size       = IMG_SIZE,
        batch_size       = BATCH_SIZE,
        label_mode       = "binary",
    )

    augment   = tf.keras.Sequential([
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
    ])
    normalize = tf.keras.Sequential([layers.Rescaling(1.0 / 255)])

    class_names = train_ds.class_names  # сохраняем до prefetch

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (normalize(x, training=False), y),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model() -> Model:
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    x   = base.output
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)  # бинарная классификация
    return Model(inputs=base.input, outputs=out)


if __name__ == "__main__":
    print("Обучение валидатора стопы...")
    train_ds, val_ds, class_names = get_datasets()
    print(f"Классы: {class_names}")

    model = build_model()
    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-3),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "validate_best.keras"),
            save_best_only=True, monitor="val_accuracy"
        ),
    ]

    print("\n--- Обучение головы ---")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Fine-tuning верхних слоёв
    print("\n--- Fine-tuning ---")
    for layer in model.layers[-30:]:
        layer.trainable = True
    for layer in model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-5),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              callbacks=[EarlyStopping(patience=4, restore_best_weights=True)])

    model.save(os.path.join(MODELS_DIR, "validate.keras"))

    # Сохраняем маппинг классов
    class_map = {str(i): name for i, name in enumerate(class_names)}
    with open(os.path.join(MODELS_DIR, "validate_classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n✓ Валидатор: val_accuracy = {acc:.3f}")
