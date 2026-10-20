"""
generate_validation_dataset.py
Класс "стопа"    — синтетика + реальные фото из папки real_feet/
Класс "не_стопа" — реальные фото CIFAR-10
"""

import os
import random
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR  = "dataset/validate"
REAL_DIR    = "real_feet"
IMG_SIZE    = (224, 224)
SAMPLES     = 500

random.seed(99)
np.random.seed(99)

os.makedirs(f"{OUTPUT_DIR}/стопа",    exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/не_стопа", exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)


def generate_foot() -> Image.Image:
    W, H = IMG_SIZE
    bg   = random.randint(210, 255)
    img  = Image.new("RGB", (W, H), (bg, bg - 5, bg - 8))
    draw = ImageDraw.Draw(img)
    pad   = random.uniform(0.08, 0.16)
    cx    = random.uniform(0.44, 0.56)
    fh    = 1.0 - 2 * pad
    fw    = random.uniform(0.30, 0.52)
    mid_w = fw * random.uniform(0.25, 0.75)
    heel_off = random.uniform(-0.25, 0.25)
    heel_cx  = cx + heel_off * fw * 0.5
    y_top  = pad
    y_mt   = pad + fh * 0.28
    y_mb   = pad + fh * 0.62
    y_heel = 1.0 - pad - fh * 0.06
    def pt(nx, ny): return (int(nx * W), int(ny * H))
    pts = [
        pt(cx, y_top), pt(cx + fw, y_mt), pt(cx + mid_w, y_mb),
        pt(heel_cx + fw*0.5, y_heel), pt(heel_cx, 1.0 - pad),
        pt(heel_cx - fw*0.5, y_heel), pt(cx - mid_w, y_mb), pt(cx - fw*0.9, y_mt),
    ]
    noise = int(W * 0.012)
    pts = [(x + random.randint(-noise, noise), y + random.randint(-noise, noise)) for x, y in pts]
    sr, sg, sb = random.randint(170,225), random.randint(130,180), random.randint(100,145)
    draw.polygon(pts, fill=(sr, sg, sb))
    toe_y = int(y_top * H) + random.randint(2, 8)
    tw = int(fw * W * 1.5)
    ts = int((cx - fw * 0.75) * W)
    for i in range(5):
        tx = ts + int(i * tw / 5)
        th = int(H * 0.05 * (1.0 - abs(i - 0.7) * 0.2))
        tr = int(tw / 13)
        draw.ellipse([tx-tr, toe_y-th, tx+tr, toe_y+2], fill=(sr-12, sg-12, sb-12))
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
    arr = np.array(img).astype(np.float32) * random.uniform(0.8, 1.2)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def augment_real(img: Image.Image) -> Image.Image:
    """Аугментация реального фото"""
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    # Случайная яркость
    arr *= random.uniform(0.75, 1.25)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Случайный поворот
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, fillcolor=(240, 235, 230))
    # Случайный flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


if __name__ == "__main__":
    print("Генерация датасета для валидатора...")

    # Реальные фото стоп из папки real_feet/
    real_files = [f for f in os.listdir(REAL_DIR)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if real_files:
        print(f"\nНайдено {len(real_files)} реальных фото стоп — генерируем аугментации...")
        aug_per_photo = max(1, 80 // len(real_files))  # ~80 аугментаций с каждого фото
        real_count = 0
        for fname in real_files:
            img = Image.open(os.path.join(REAL_DIR, fname)).convert("RGB")
            for j in range(aug_per_photo):
                aug = augment_real(img)
                aug.save(f"{OUTPUT_DIR}/стопа/real_{real_count:04d}.jpg", quality=85)
                real_count += 1
        print(f"  Реальных аугментаций: {real_count}")
        synth_count = max(0, SAMPLES - real_count)
    else:
        print(f"\n⚠ Папка {REAL_DIR}/ пуста — используем только синтетику.")
        print(f"  Для лучшего результата добавь 5–10 своих фото стоп в папку {REAL_DIR}/")
        synth_count = SAMPLES

    # Синтетические стопы
    print(f"\nГенерация синтетических стоп: {synth_count}...")
    for i in range(synth_count):
        generate_foot().save(f"{OUTPUT_DIR}/стопа/synth_{i:04d}.jpg", quality=85)
        if (i + 1) % 100 == 0:
            print(f"  синтетика: {i+1}/{synth_count}")

    # Не стопы из CIFAR-10
    print("\nЗагружаем CIFAR-10...")
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    all_imgs = np.concatenate([x_train, x_test], axis=0)
    indices  = np.random.choice(len(all_imgs), SAMPLES, replace=False)
    for i, idx in enumerate(indices):
        img = Image.fromarray(all_imgs[idx]).resize(IMG_SIZE, Image.LANCZOS)
        img.save(f"{OUTPUT_DIR}/не_стопа/cifar_{i:04d}.jpg", quality=85)
        if (i + 1) % 100 == 0:
            print(f"  не_стопа: {i+1}/{SAMPLES}")

    total = len(os.listdir(f"{OUTPUT_DIR}/стопа"))
    print(f"\nГотово!")
    print(f"  стопа:    {total} изображений")
    print(f"  не_стопа: {SAMPLES} изображений")

    if not real_files:
        print(f"\nСОВЕТ: добавь свои фото стоп в папку '{REAL_DIR}/' и")
        print(f"  запусти скрипт снова — точность валидатора вырастет.")