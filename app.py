"""
app.py — FastAPI-эндпоинт, связывает analyzer и matcher
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import shutil
import traceback
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from analyzer import FootAnalyzer
from matcher  import ShoeMatcher

app = FastAPI(
    title="Сервис подбора обуви",
    description="Анализирует фото стопы и возвращает подходящие модели обуви",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "shoe_service"),
}

analyzer = FootAnalyzer()
matcher  = ShoeMatcher(DB_CONFIG)

TMP_DIR = os.path.join(os.getcwd(), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)


@app.post("/api/recommend")
async def recommend(
    file:     UploadFile    = File(...,  description="Фото стопы (jpg/png)"),
    gender:   Optional[str] = Form(None, description="мужская | женская | унисекс"),
    category: Optional[str] = Form(None, description="Бег | Повседневные | Тренировки | Ходьба"),
    top_n:    Optional[int] = Form(5,    description="Количество рекомендаций"),
):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Поддерживаются только JPG и PNG")

    file_id   = uuid.uuid4().hex
    tmp_input = os.path.join(TMP_DIR, f"{file_id}_input.jpg")
    tmp_debug = os.path.join(TMP_DIR, f"{file_id}_debug.jpg")

    try:
        contents = await file.read()
        with open(tmp_input, "wb") as f:
            f.write(contents)
        if not os.path.exists(tmp_input):
            raise HTTPException(status_code=500, detail=f"Не удалось записать файл: {tmp_input}")

        # Валидация — проверяем что на фото стопа
        is_foot, foot_conf = analyzer.validate_foot(tmp_input)
        if not is_foot:
            raise HTTPException(
                status_code=422,
                detail=f"На фото не обнаружена стопа (уверенность: {foot_conf:.0%}). "                        "Сфотографируйте подошву стопы снизу на светлом фоне."
            )

        # Анализируем фото
        params = analyzer.analyze(tmp_input, save_debug=tmp_debug)

        # Подбираем обувь
        matches = matcher.find_matches(
            params   = params,
            gender   = gender,
            category = category,
            top_n    = top_n,
        )

        return {
            "foot_params": {
                "arch_support":   params.arch_support,
                "pronation_type": params.pronation_type,
                "width_type":     params.width_type,
                "confidence":     params.confidence,
            },
            "debug_image_id": file_id,
            "recommendations": [
                {
                    "rank":          i + 1,
                    "shoe_id":       m.shoe_id,
                    "model":         m.model_name,
                    "brand":         m.brand,
                    "category":      m.category,
                    "gender":        m.gender,
                    "description":   m.description,
                    "score":         m.score,
                    "match_details": m.match_details,
                }
                for i, m in enumerate(matches)
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        # Выводим полный traceback в терминал для отладки
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_input):
            os.remove(tmp_input)


@app.get("/api/debug/{file_id}", response_class=FileResponse)
async def get_debug_image(file_id: str):
    path = os.path.join(TMP_DIR, f"{file_id}_debug.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Изображение не найдено или устарело")
    return FileResponse(path, media_type="image/jpeg", filename="debug.jpg")




@app.get("/api/catalog")
async def catalog(
    brand:    Optional[str] = None,
    category: Optional[str] = None,
    gender:   Optional[str] = None,
):
    """Возвращает каталог обуви с ортопедическими свойствами."""
    import mysql.connector
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT
                s.id, s.model_name, s.gender, s.description,
                b.name  AS brand,
                c.name  AS category,
                p.arch_support, p.pronation_type,
                p.width_type, p.cushioning_level,
                p.surface_type, p.is_ortho_friendly
            FROM shoes s
            JOIN brands          b ON b.id = s.brand_id
            JOIN categories      c ON c.id = s.category_id
            JOIN shoe_properties p ON p.shoe_id = s.id
            WHERE 1=1
        """
        args = []
        if brand:
            query += " AND b.name = %s"
            args.append(brand)
        if category:
            query += " AND c.name = %s"
            args.append(category)
        if gender:
            query += " AND (s.gender = %s OR s.gender = 'унисекс')"
            args.append(gender)

        query += " ORDER BY b.name, s.model_name"
        cursor.execute(query, args)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"shoes": rows, "total": len(rows)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/catalog/filters")
async def catalog_filters():
    """Возвращает доступные значения фильтров."""
    import mysql.connector
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT DISTINCT name FROM brands ORDER BY name")
        brands = [r["name"] for r in cursor.fetchall()]
        cursor.execute("SELECT DISTINCT name FROM categories ORDER BY name")
        categories = [r["name"] for r in cursor.fetchall()]
        cursor.close()
        conn.close()
        return {"brands": brands, "categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/health")
async def health():
    return {"status": "ok"}
