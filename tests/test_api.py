import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

HAS_MODEL = os.path.exists("./data/scibert_quality_model") or os.path.exists("/app/data/scibert_quality_model")


class TestHealth:
    """Тесты health check"""

    def test_health_endpoint(self):
        """Проверка что health возвращает 200 и правильные данные"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["version"] == "1.0.0"
        print("✅ Health check passed")


class TestPredict:
    """Тесты предсказаний"""

    def test_predict_text(self):
        """Проверка предсказания научного текста"""
        text = "This paper presents a new method for optimizing neural networks. The proposed approach can be applied to computer-aided problem solving."
        
        response = client.post("/predict", json={"text": text})
        
        if HAS_MODEL:
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == text
            assert data["quality"] in ["good", "bad"]
            assert 0 <= data["confidence"] <= 1
            assert data["processing_time_ms"] > 0
            print(f"✅ Prediction: {data['quality']} with confidence {data['confidence']:.2%}")
        else:
            print("⚠️ Skipping prediction test (no model in CI/CD)")
    
    def test_predict_too_short_text(self):
        """Проверка слишком короткого текста (ошибка валидации)"""
        response = client.post("/predict", json={"text": "short"})
        assert response.status_code == 422
        print("✅ Too short text correctly rejected (422)")
    
    def test_predict_empty_text(self):
        """Проверка пустого текста (ошибка валидации)"""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422
        print("✅ Empty text correctly rejected (422)")


class TestMetrics:
    """Тесты метрик Prometheus"""

    def test_metrics_endpoint(self):
        """Проверка что метрики доступны"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.text
        assert "predict_requests_total" in data
        assert "model_loaded" in data
        print("✅ Metrics endpoint works")


class TestRoot:
    """Тесты корневого эндпоинта"""

    def test_root_endpoint(self):
        """Проверка что корневой эндпоинт возвращает информацию"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        print("✅ Root endpoint works")


class TestResponseTime:
    """Тесты времени ответа"""

    def test_response_time_under_2_seconds(self):
        """Проверка времени ответа (только если есть модель)"""
        if not HAS_MODEL:
            print("⚠️ Skipping response time test (no model in CI/CD)")
            return
        
        text = "Scientific text for testing the response time of a classification model."
        
        start = time.time()
        response = client.post("/predict", json={"text": text})
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0
        print(f"✅ Response time: {elapsed:.2f} seconds")