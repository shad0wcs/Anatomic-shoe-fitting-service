"""
matcher.py — блок подбора обуви по параметрам стопы
Принимает FootParams, возвращает ранжированный список моделей из БД
"""

import mysql.connector
from dataclasses import dataclass
from typing import List
from analyzer import FootParams


@dataclass
class ShoeMatch:
    shoe_id:         int
    model_name:      str
    brand:           str
    category:        str
    gender:          str
    description:     str
    score:           float   # итоговый скоринг 0–100
    match_details:   dict    # детализация по каждому параметру


class ShoeMatcher:
    """
    Подбирает обувь из БД по параметрам стопы.
    Используется взвешенный скоринг: каждый параметр вносит свой вклад.
    """

    # Веса параметров (в сумме 100)
    WEIGHTS = {
        "arch_support":      35,   # тип свода — самый важный
        "pronation_type":    30,   # пронация
        "width_type":        20,   # ширина
        "cushioning_level":  15,   # амортизация (вторичный критерий)
    }

    # Матрица совместимости свода пользователя → рекомендуемый arch_support обуви
    ARCH_COMPAT = {
        "отсутствует": {"высокая": 1.0, "средняя": 0.5, "низкая": 0.2, "отсутствует": 0.0},
        "низкая":      {"высокая": 0.8, "средняя": 1.0, "низкая": 0.6, "отсутствует": 0.1},
        "средняя":     {"высокая": 0.4, "средняя": 1.0, "низкая": 0.7, "отсутствует": 0.3},
        "высокая":     {"высокая": 1.0, "средняя": 0.6, "низкая": 0.3, "отсутствует": 0.1},
    }

    # Матрица совместимости пронации
    PRONATION_COMPAT = {
        "гиперпронация": {"гиперпронация": 1.0, "нейтральная": 0.5, "супинация": 0.0},
        "нейтральная":   {"нейтральная":   1.0, "гиперпронация": 0.4, "супинация": 0.4},
        "супинация":     {"супинация":     1.0, "нейтральная": 0.5, "гиперпронация": 0.0},
    }

    # Матрица совместимости ширины
    WIDTH_COMPAT = {
        "узкая":          {"узкая": 1.0, "стандартная": 0.6, "широкая": 0.2, "очень широкая": 0.0},
        "стандартная":    {"стандартная": 1.0, "узкая": 0.5, "широкая": 0.7, "очень широкая": 0.3},
        "широкая":        {"широкая": 1.0, "очень широкая": 0.8, "стандартная": 0.4, "узкая": 0.0},
        "очень широкая":  {"очень широкая": 1.0, "широкая": 0.7, "стандартная": 0.2, "узкая": 0.0},
    }

    # Предпочтительный уровень амортизации при проблемах со сводом
    CUSHION_PREF = {
        "отсутствует": "максимальная",
        "низкая":      "максимальная",
        "средняя":     "умеренная",
        "высокая":     "умеренная",
    }

    CUSHION_COMPAT = {
        "максимальная": {"максимальная": 1.0, "умеренная": 0.6, "минимальная": 0.1},
        "умеренная":    {"умеренная":    1.0, "максимальная": 0.7, "минимальная": 0.5},
        "минимальная":  {"минимальная":  1.0, "умеренная": 0.6, "максимальная": 0.3},
    }

    def __init__(self, db_config: dict):
        self.db_config = db_config

    # Основной метод

    def find_matches(
        self,
        params: FootParams,
        gender: str = None,       # 'мужская' | 'женская' | 'унисекс' | None
        category: str = None,     # 'Бег' | 'Повседневные' | ... | None
        top_n: int = 5,
        min_score: float = 30.0,  # отсекаем явно несовместимые модели
    ) -> List[ShoeMatch]:
        """
        Возвращает топ-N моделей, отсортированных по убыванию скоринга.
        """
        shoes = self._fetch_shoes(gender, category)
        results = []

        for shoe in shoes:
            score, details = self._score_shoe(params, shoe)
            if score >= min_score:
                results.append(ShoeMatch(
                    shoe_id       = shoe["id"],
                    model_name    = shoe["model_name"],
                    brand         = shoe["brand"],
                    category      = shoe["category"],
                    gender        = shoe["gender"],
                    description   = shoe["description"] or "",
                    score         = round(score, 1),
                    match_details = details,
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]

    # Загрузка моделей из БД

    def _fetch_shoes(self, gender: str = None, category: str = None) -> list:
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT
                s.id,
                s.model_name,
                s.gender,
                s.description,
                b.name  AS brand,
                c.name  AS category,
                p.arch_support,
                p.pronation_type,
                p.width_type,
                p.cushioning_level,
                p.is_ortho_friendly
            FROM shoes s
            JOIN brands          b ON b.id = s.brand_id
            JOIN categories      c ON c.id = s.category_id
            JOIN shoe_properties p ON p.shoe_id = s.id
            WHERE 1=1
        """
        args = []

        if gender:
            query += " AND (s.gender = %s OR s.gender = 'унисекс')"
            args.append(gender)

        if category:
            query += " AND c.name = %s"
            args.append(category)

        cursor.execute(query, args)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()
        return rows

    # Скоринг одной модели
    def _score_shoe(self, params: FootParams, shoe: dict) -> tuple[float, dict]:
        """
        Возвращает (итоговый балл 0–100, детализация по параметрам).
        """
        w = self.WEIGHTS

        # Совместимость по своду
        arch_score = self.ARCH_COMPAT \
            .get(params.arch_support, {}) \
            .get(shoe["arch_support"], 0.0)

        # Совместимость по пронации
        pron_score = self.PRONATION_COMPAT \
            .get(params.pronation_type, {}) \
            .get(shoe["pronation_type"], 0.0)

        # Совместимость по ширине
        width_score = self.WIDTH_COMPAT \
            .get(params.width_type, {}) \
            .get(shoe["width_type"], 0.0)

        # Совместимость по амортизации
        preferred_cushion = self.CUSHION_PREF.get(params.arch_support, "умеренная")
        cush_score = self.CUSHION_COMPAT \
            .get(preferred_cushion, {}) \
            .get(shoe["cushioning_level"], 0.0)

        # Бонус: модель явно помечена как ортопедически совместимая
        ortho_bonus = 5.0 if (
            shoe["is_ortho_friendly"]
            and params.arch_support in ("отсутствует", "низкая")
        ) else 0.0

        total = (
            arch_score  * w["arch_support"]    +
            pron_score  * w["pronation_type"]  +
            width_score * w["width_type"]      +
            cush_score  * w["cushioning_level"] +
            ortho_bonus
        )

        details = {
            "свод":       round(arch_score  * w["arch_support"],    1),
            "пронация":   round(pron_score  * w["pronation_type"],  1),
            "ширина":     round(width_score * w["width_type"],      1),
            "амортизация":round(cush_score  * w["cushioning_level"],1),
            "ортобонус":  ortho_bonus,
        }

        return min(total, 100.0), details
