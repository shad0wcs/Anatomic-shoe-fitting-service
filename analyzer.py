"""
analyzer.py — анализ стопы через CNN (MobileNetV2)
"""

import os
import json
import numpy as np
import cv2
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
import keras


@dataclass
class FootParams:
    arch_support:   str     # отсутствует | низкая | средняя | высокая
    pronation_type: str     # нейтральная | гиперпронация | супинация
    width_type:     str     # узкая | стандартная | широкая | очень широкая
    confidence:     float   # 0.0 – 1.0 (среднее softmax-максимумов)


class FootAnalyzer:

    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    IMG_SIZE   = (224, 224)

    FONT_PATHS = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    def __init__(self):
        self._font_path = self._find_font()
        self._models    = {}
        self._classes   = {}
        self._load_models()

    # Загрузка моделей

    def _load_models(self):
        for task in ["arch", "pronation", "width"]:
            model_path  = os.path.join(self.MODELS_DIR, f"{task}.keras")
            classes_path = os.path.join(self.MODELS_DIR, f"{task}_classes.json")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Модель не найдена: {model_path}\n"
                    f"Запустите train_model.py для обучения."
                )

            self._models[task]  = keras.models.load_model(model_path)
            with open(classes_path, encoding="utf-8") as f:
                self._classes[task] = json.load(f)

        print("Модели загружены успешно")

    # Основной метод

    def validate_foot(self, image_path: str) -> tuple[bool, float]:
        """
        Проверяет что на фото изображена стопа.
        Возвращает (является_стопой, уверенность).
        """
        if "validate" not in self._models:
            return True, 1.0

        with open(image_path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False, 0.0

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize(self.IMG_SIZE, Image.LANCZOS)
        arr     = np.array(img_pil).astype(np.float32) / 255.0
        arr     = np.expand_dims(arr, axis=0)

        prob = float(self._models["validate"].predict(arr, verbose=0)[0][0])

        # Определяем какой класс соответствует стопе
        classes = self._classes["validate"]
        foot_idx = next((k for k, v in classes.items() if "стопа" in v and "не" not in v), "1")
        is_foot = (foot_idx == "1" and prob >= 0.5) or (foot_idx == "0" and prob < 0.5)
        confidence = prob if foot_idx == "1" else 1.0 - prob

        return is_foot, round(confidence, 2)

    def analyze(self, image_path: str, save_debug: str = None) -> FootParams:
        # cv2.imread не поддерживает кириллицу на Windows
        with open(image_path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        img_bgr = cv2.resize(img_bgr, (640, 480))

        # Предсказания CNN
        arch_label,      arch_conf      = self._predict("arch",      img_bgr)
        pronation_label, pronation_conf = self._predict("pronation", img_bgr)
        width_label,     width_conf     = self._predict("width",     img_bgr)

        # Уверенность = среднее трёх softmax-максимумов
        confidence = round(float(np.mean([arch_conf, pronation_conf, width_conf])), 2)

        params = FootParams(
            arch_support   = arch_label,
            pronation_type = pronation_label,
            width_type     = width_label,
            confidence     = confidence,
        )

        if save_debug:
            self._save_visualization(
                img_bgr, params,
                confs = {
                    "arch":      arch_conf,
                    "pronation": pronation_conf,
                    "width":     width_conf,
                },
                save_path = save_debug,
            )

        return params


    # Предсказание одного классификатора


    def _predict(self, task: str, img_bgr: np.ndarray) -> tuple[str, float]:
        """
        Возвращает (метка_класса, уверенность 0..1).
        """
        # Предобработка под MobileNetV2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize(self.IMG_SIZE, Image.LANCZOS)
        arr     = np.array(img_pil).astype(np.float32) / 255.0
        arr     = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

        probs   = self._models[task].predict(arr, verbose=0)[0]  # (num_classes,)
        idx     = int(np.argmax(probs))
        label   = self._classes[task][str(idx)]
        conf    = float(probs[idx])

        return label, conf

    # Визуализация — панель с барчартами уверенности

    def _save_visualization(
        self,
        img_bgr:   np.ndarray,
        params:    FootParams,
        confs:     dict,
        save_path: str,
    ):
        vis = img_bgr.copy()

        # Панель снизу
        panel_h = 130
        panel   = np.zeros((panel_h, vis.shape[1], 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        result  = np.vstack([vis, panel])

        img_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)

        try:
            font_main  = ImageFont.truetype(self._font_path, 14) if self._font_path else ImageFont.load_default()
            font_small = ImageFont.truetype(self._font_path, 11) if self._font_path else ImageFont.load_default()
        except Exception:
            font_main = font_small = ImageFont.load_default()

        H      = result.shape[0]
        panel_y0 = H - panel_h
        W      = result.shape[1]

        # Заголовок
        draw.text((12, panel_y0 + 8), "Анализ стопы (CNN)", font=font_main, fill=(200, 200, 200))

        # Три блока: свод / пронация / ширина
        items = [
            ("Свод",     params.arch_support,   confs["arch"],      (50, 200, 100)),
            ("Пронация", params.pronation_type,  confs["pronation"], (100, 160, 240)),
            ("Ширина",   params.width_type,      confs["width"],     (240, 160, 60)),
        ]

        col_w = W // 3
        bar_h = 12
        bar_max_w = col_w - 24

        for i, (label, value, conf, color) in enumerate(items):
            lx = i * col_w + 12
            ly = panel_y0 + 34

            draw.text((lx, ly),      label, font=font_small, fill=(160, 160, 160))
            draw.text((lx, ly + 16), value, font=font_main,  fill=(255, 255, 255))

            # Бар уверенности
            bar_y = ly + 38
            bar_w = int(bar_max_w * conf)
            draw.rectangle([lx, bar_y, lx + bar_max_w, bar_y + bar_h],
                           outline=(80, 80, 80), width=1)
            if bar_w > 0:
                draw.rectangle([lx + 1, bar_y + 1, lx + bar_w, bar_y + bar_h - 1],
                               fill=color)
            draw.text((lx, bar_y + bar_h + 4), f"{conf:.0%}", font=font_small, fill=color)

        # Общая уверенность
        draw.text(
            (W - 130, panel_y0 + 8),
            f"Точность: {params.confidence:.0%}",
            font=font_main, fill=(200, 200, 200)
        )

        result_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".jpg", result_bgr)
        if success:
            with open(save_path, "wb") as f:
                f.write(buf.tobytes())


    def _find_font(self) -> str:
        for p in self.FONT_PATHS:
            if os.path.exists(p):
                return p
        return None
