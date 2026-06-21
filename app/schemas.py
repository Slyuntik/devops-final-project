from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    quality: str
    confidence: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: Optional[bool] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
