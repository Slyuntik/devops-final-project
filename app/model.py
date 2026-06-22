import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SciBERTQualityClassifier:
    """
    Классификатор качества научных текстов на основе SciBERT.
    Бинарная классификация: good (хорошее качество) vs bad (плохое качество).
    """
    
    def __init__(self, model_path: str):
        """
        Загружает модель и токенизатор из папки.
        
        Args:
            model_path: путь к папке с model.safetensors
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        logger.info(f"Загрузка модели из {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        self.model.eval()
        
        logger.info(f"Модель загружена на {self.device}")
    
    def predict(self, text: str, max_length: int = 256) -> tuple:
        """
        Предсказание качества одного текста.
        
        Args:
            text: научно-технический текст (для обучения использовались аннотации)
            max_length: максимальная длина токенов (для обучения использовалось 256)
        
        Returns:
            (label, confidence): label = "good" или "bad", confidence = 0..1
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        label = "good" if predicted_class == 1 else "bad"
        confidence = probs[0, predicted_class].item()
        
        return label, confidence


_model_instance = None

def get_model(model_path: str = None) -> SciBERTQualityClassifier:
    """
    Возвращает загруженную модель.
    При первом вызове загружает её, при последующих возвращает уже загруженную.
    """
    global _model_instance
    if _model_instance is None:
        path = model_path or os.getenv("MODEL_PATH")
        
        if not path:
            if os.path.exists("./data/scibert_quality_model"):
                path = "./data/scibert_quality_model"
            else:
                path = "/app/data/scibert_quality_model"
        
        logger.info(f"Загрузка модели из {path}")
        _model_instance = SciBERTQualityClassifier(path)
    return _model_instance