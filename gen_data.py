"""
gen_data.py — генератор синтетических плантограмм
Создаёт датасет изображений стоп с разными параметрами для обучения CNN
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json


# Настройки
OUTPUT_DIR  = "dataset"
IMG_SIZE    = (224, 224)
SAMPLES     = 500

ARCH_CLASSES     = ["отсутствует", "низкая", "средняя", "высокая"]
PRONATION_CLASSES = ["гиперпронация", "нейтральная", "супинация"]
WIDTH_CLASSES    = ["очень широкая", "широкая", "стандартная", "узкая"]

random.seed(42)
np.random.seed(42)



# Параметры формы стопы по классу

# arch_index: доля среднего отдела (выше = плоскостопие)
# Увеличены диапазоны — чёткое разделение классов
ARCH_PARAMS = {
    "отсутствует": dict(arch_index=(0.36, 0.46), arch_noise=0.01),
    "низкая":      dict(arch_index=(0.28, 0.35), arch_noise=0.01),
    "средняя":     dict(arch_index=(0.19, 0.27), arch_noise=0.01),
    "высокая":     dict(arch_index=(0.06, 0.17), arch_noise=0.01),
}

PRONATION_PARAMS = {
    "гиперпронация": dict(heel_offset=(0.18, 0.30)),   # сильный завал внутрь
    "нейтральная":   dict(heel_offset=(-0.04, 0.04)),  # строго по центру
    "супинация":     dict(heel_offset=(-0.30, -0.18)), # сильный завал наружу
}

WIDTH_PARAMS = {
    "очень широкая": dict(forefoot_ratio=(0.50, 0.62)),
    "широкая":       dict(forefoot_ratio=(0.42, 0.49)),
    "стандартная":   dict(forefoot_ratio=(0.34, 0.41)),
    "узкая":         dict(forefoot_ratio=(0.22, 0.32)),
}


# Генератор силуэта стопы
def generate_foot(
    arch_class: str,
    pronation_class: str,
    width_class: str,
    size: tuple = IMG_SIZE,
) -> Image.Image:
    """
    Рисует синтетический силуэт стопы (вид снизу) с заданными параметрами.
    Возвращает PIL Image в градациях серого на белом фоне.
    """
    W, H = size
    img  = Image.new("RGB", (W, H), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Случайные параметры из диапазонов класса
    ap = ARCH_PARAMS[arch_class]
    pp = PRONATION_PARAMS[pronation_class]
    wp = WIDTH_PARAMS[width_class]

    arch_idx     = random.uniform(*ap["arch_index"])
    heel_offset  = random.uniform(*pp["heel_offset"])
    fore_ratio   = random.uniform(*wp["forefoot_ratio"])

    # Добавляем шум освещения
    bg_val = random.randint(220, 255)
    img    = Image.new("RGB", (W, H), color=(bg_val, bg_val - 5, bg_val - 10))
    draw   = ImageDraw.Draw(img)

    # Опорные точки стопы (нормализованы в 0..1, затем масштабируются)
    # Стопа ориентирована вертикально: пальцы сверху, пятка снизу
    pad   = 0.12   # отступ от краёв
    cx    = 0.5    # центр по X

    foot_top    = pad              # верх (пальцы)
    foot_bottom = 1.0 - pad       # низ (пятка)
    foot_height = foot_bottom - foot_top

    # Ширина в трёх зонах
    fore_w  = fore_ratio * foot_height * 0.55        # передний отдел
    mid_w   = fore_w * (1.0 - arch_idx) * 1.1       # средний (свод) — уже при высоком своде
    heel_w  = fore_w * 0.62                          # пятка

    # Смещение пятки (пронация)
    heel_cx = cx + heel_offset * heel_w

    # Y-координаты зон
    y_fore  = foot_top + foot_height * 0.18
    y_mid_t = foot_top + foot_height * 0.35
    y_mid_b = foot_top + foot_height * 0.65
    y_heel  = foot_bottom - foot_height * 0.08

    def pt(nx, ny):
        return (int(nx * W), int(ny * H))

    # Контур стопы из опорных точек (полигон)
    points = [
        pt(cx,              foot_top),            # кончик среднего пальца
        pt(cx + fore_w,     y_fore),              # внешний край переднего отдела
        pt(cx + fore_w * 0.9, y_mid_t),           # внешний переход к своду
        pt(cx + mid_w * 0.5, y_mid_b),            # внешний свод
        pt(heel_cx + heel_w, y_heel),             # внешний край пятки
        pt(heel_cx,          foot_bottom),        # нижняя точка пятки
        pt(heel_cx - heel_w, y_heel),             # внутренний край пятки
        pt(cx - mid_w * 0.5, y_mid_b),            # внутренний свод
        pt(cx - fore_w * 0.85, y_mid_t),          # внутренний переход
        pt(cx - fore_w * 0.95, y_fore),           # внутренний край переднего отдела
    ]

    # Добавляем случайный шум к точкам (имитация живой стопы)
    noise_px = int(W * 0.015)
    noisy_points = [
        (x + random.randint(-noise_px, noise_px),
         y + random.randint(-noise_px, noise_px))
        for x, y in points
    ]

    # Рисуем тело стопы
    skin_r = random.randint(180, 220)
    skin_g = random.randint(140, 180)
    skin_b = random.randint(110, 150)
    draw.polygon(noisy_points, fill=(skin_r, skin_g, skin_b))

    # Добавляем пальцы (5 маленьких эллипсов сверху)
    toe_y   = int(foot_top * H) + random.randint(2, 6)
    toe_w_total = int(fore_w * W * 1.6)
    toe_start   = int((cx - fore_w * 0.8) * W)
    for i in range(5):
        tx   = toe_start + int(i * toe_w_total / 5)
        # Пальцы разной длины (большой самый длинный)
        toe_h = int(H * 0.055 * (1.0 - abs(i - 0.8) * 0.18))
        toe_r = int(toe_w_total / 12)
        draw.ellipse(
            [tx - toe_r, toe_y - toe_h, tx + toe_r, toe_y + 2],
            fill=(skin_r - 10, skin_g - 10, skin_b - 10)
        )

    # Мягкое размытие — имитация фотографии
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    # Случайная яркость/контраст
    factor = random.uniform(0.85, 1.15)
    arr = np.array(img).astype(np.float32) * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


# Генерация датасета
def generate_dataset(output_dir: str = OUTPUT_DIR, samples: int = SAMPLES):
    """
    Создаёт три поддатасета:
      dataset/arch/       — классификация свода
      dataset/pronation/  — классификация пронации
      dataset/width/      — классификация ширины
    """
    stats = {}

    tasks = [
        ("arch",      ARCH_CLASSES,      lambda c: generate_foot(arch_class=c,      pronation_class="нейтральная", width_class="стандартная")),
        ("pronation", PRONATION_CLASSES, lambda c: generate_foot(arch_class="средняя", pronation_class=c,          width_class="стандартная")),
        ("width",     WIDTH_CLASSES,     lambda c: generate_foot(arch_class="средняя", pronation_class="нейтральная", width_class=c)),
    ]

    for task_name, classes, gen_fn in tasks:
        task_dir = os.path.join(output_dir, task_name)
        stats[task_name] = {}
        total = len(classes) * samples

        print(f"\n{'='*50}")
        print(f"Генерация: {task_name} ({total} изображений)")

        count = 0
        for cls in classes:
            cls_dir = os.path.join(task_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)

            for i in range(samples):
                img = gen_fn(cls)
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                fname = os.path.join(cls_dir, f"{cls}_{i:04d}.jpg")
                img.save(fname, quality=85)
                count += 1

                if count % 100 == 0:
                    print(f"  [{count}/{total}] {cls} — {i+1}/{samples}")

            stats[task_name][cls] = samples
            print(f"  ✓ {cls}: {samples} изображений")

    # Сохраняем мета-информацию
    meta = {
        "img_size":  list(IMG_SIZE),
        "samples_per_class": samples,
        "tasks": {
            "arch":      ARCH_CLASSES,
            "pronation": PRONATION_CLASSES,
            "width":     WIDTH_CLASSES,
        },
        "stats": stats,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total_imgs = sum(len(c) * samples for _, c, _ in tasks)
    print(f"\n{'='*50}")
    print(f"Готово! Всего изображений: {total_imgs}")
    print(f"Датасет сохранён в: {os.path.abspath(output_dir)}")
    return stats


if __name__ == "__main__":
    generate_dataset()
