from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime


class PredictRequest(BaseModel):
    """
    Что клиент отправляет на /predict
    """
    text: str = Field(..., min_length=10, max_length=10000, description="Текст для классификации")
    
    @validator('text')
    def text_not_empty(cls, v: str) -> str:
        """Проверяем, что текст не состоит из пробелов"""
        if not v.strip():
            raise ValueError('Текст не может быть пустым')
        return v.strip()
    
    @validator('text')
    def text_not_too_short(cls, v: str) -> str:
        """Минимальная длина для осмысленной классификации"""
        if len(v) < 20:
            raise ValueError('Текст слишком короткий (минимум 20 символов)')
        return v


class PredictResponse(BaseModel):
    """
    Что сервер возвращает на запрос
    """
    text: str = Field(..., description="Оригинальный текст")
    quality: str = Field(..., description="good (хорошее качество) или bad (плохое качество)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность от 0 до 1")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время запроса")


class HealthResponse(BaseModel):
    """
    Ответ на health check (для Docker и Kubernetes)
    """
    status: str = Field(..., description="healthy / degraded / unhealthy")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    version: str = Field(..., description="Версия API")


class ErrorResponse(BaseModel):
    """
    Стандартный ответ при ошибке
    """
    detail: str = Field(..., description="Описание ошибки")
    timestamp: datetime = Field(default_factory=datetime.now)