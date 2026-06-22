from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time
import logging
from contextlib import asynccontextmanager
import asyncpg
import os

from app.schemas import PredictRequest, PredictResponse, HealthResponse, ErrorResponse
from app.model import get_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

requests_total = Counter(
    'predict_requests_total',
    'Total number of prediction requests',
    ['status']
)

prediction_duration = Histogram(
    'prediction_duration_seconds',
    'Time spent processing prediction',
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

text_length = Histogram(
    'input_text_length_characters',
    'Length of input text in characters',
    buckets=[50, 100, 200, 500, 1000, 2000, 5000]
)

model_status = Gauge(
    'model_loaded',
    'Whether the ML model is loaded (1) or not (0)'
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения.
    Выполняется ДО того как сервер начнет принимать запросы.
    """
    logger.info("Запуск приложения - загружаем модель SciBERT...")
    model = get_model()
    model_status.set(1)
    logger.info("Модель успешно загружена, готова к приему запросов")

    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:changeme123@db:5432/scibert_quality')
    try:
        app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)
        logger.info("Подключение к PostgreSQL установлено")
    except Exception as e:
        logger.warning(f"Не удалось подключиться к БД: {e}")
        app.state.db_pool = None
    
    yield
    
    logger.info("Остановка приложения...")
    model_status.set(0)
    if app.state.db_pool:
        await app.state.db_pool.close()
        logger.info("Соединение с БД закрыто")


app = FastAPI(
    title="SciBERT Text Quality Classifier",
    description="API для бинарной классификации качества научно-технических текстов",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Проверка здоровья сервиса.
    Используется Docker для проверки, жив ли контейнер.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_status._value.get() == 1,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
async def predict(request: PredictRequest, req: Request):
    """
    Классификация качества научно-технического текста.
    
    - Возвращает "good" для качественных текстов
    - Возвращает "bad" для некачественных
    - confidence - уверенность модели (0-1)
    """
    start_time = time.time()
    text_length.observe(len(request.text))
    
    try:
        model = get_model()
        quality, confidence = model.predict(request.text)
        processing_time_ms = (time.time() - start_time) * 1000
        prediction_duration.observe(processing_time_ms / 1000)
        requests_total.labels(status="success").inc()

        if hasattr(app.state, 'db_pool') and app.state.db_pool:
            try:
                client_ip = req.client.host if req.client else "unknown"
                async with app.state.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO predictions (text, quality, confidence, processing_time_ms, ip_address)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        request.text, quality, confidence, processing_time_ms, client_ip
                    )
                logger.info(f"Предсказание сохранено в БД: {quality}")
            except Exception as db_error:
                logger.error(f"Ошибка сохранения в БД: {db_error}")
        
        return PredictResponse(
            text=request.text,
            quality=quality,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        requests_total.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Root"])
async def root():
    """
    Корневой эндпоинт с информацией об API.
    """
    return {
        "name": "SciBERT Text Quality Classifier",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Проверка здоровья (GET)",
            "/predict": "Классификация текста (POST)",
            "/metrics": "Метрики для Prometheus (GET)",
            "/docs": "Swagger документация (GET)"
        }
    }
