import os
import re
from functools import lru_cache


class SimpleTextQualityModel:
    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "/app/data/scibert_quality_model")

    def predict(self, text: str):
        text_clean = text.strip()
        words = re.findall(r"\w+", text_clean.lower())

        score = 0.5

        if len(text_clean) > 100:
            score += 0.15

        if len(words) > 20:
            score += 0.15

        scientific_words = [
            "model", "data", "analysis", "method",
            "algorithm", "result", "accuracy",
            "исследование", "модель", "данные",
            "метод", "алгоритм", "результат"
        ]

        if any(word in text_clean.lower() for word in scientific_words):
            score += 0.15

        bad_patterns = ["asdf", "qwerty", "test test", "12345"]

        if any(pattern in text_clean.lower() for pattern in bad_patterns):
            score -= 0.35

        confidence = max(0.1, min(score, 0.99))
        quality = "good" if confidence >= 0.6 else "bad"

        return quality, round(confidence, 3)


@lru_cache
def get_model():
    return SimpleTextQualityModel()
