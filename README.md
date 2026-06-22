# SciBERT Quality Classifier

[![CI/CD Pipeline](https://github.com/Slyuntik/devops-final-project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Slyuntik/devops-final-project/actions/workflows/ci-cd.yml)

Сервис для бинарной классификации качества научно-технических текстов на основе дообученной модели SciBERT.

## Описание

Проект представляет собой микросервис для автоматической оценки качества научных текстов. Модель классифицирует текст на два класса:
- **good** — качественная научно-техническая статья
- **bad** — текст низкого качества (не научный или неструктурированный)

### Основные возможности:
- REST API для классификации текстов
- Веб-интерфейс для тестирования
- Автоматическое сохранение истории предсказаний в PostgreSQL
- Мониторинг метрик через Prometheus + Grafana
- HTTPS с самоподписанным сертификатом
- CI/CD пайплайн с тестами и нагрузочным тестированием

## Технологии

| Компонент | Технология | Версия |
|-----------|------------|--------|
| Backend | FastAPI + Uvicorn | 0.115.0 / 0.24.0 |
| ML модель | SciBERT (allenai/scibert_scivocab_uncased) | fine-tuned |
| Контейнеризация | Docker + Docker Compose | 3.8 |
| Reverse Proxy | NGINX (с самоподписанным SSL) | alpine |
| База данных | PostgreSQL | 15-alpine |
| Мониторинг | Prometheus + Grafana | 2.48.0 / 10.2.0 |
| Тестирование | pytest + k6 | 7.4.3 / latest |
| CI/CD | GitHub Actions | — |

## Требования

- Docker 24.0+
- Docker Compose 2.20+
- Python 3.12 (для локальной разработки)
- Git LFS (для скачивания модели)
- 4GB+ RAM (для модели SciBERT)

## Установка и запуск

```bash
git clone https://github.com/Slyuntik/devops-project.git
cd devops-project

# Установите Git LFS (если не установлен)
brew install git-lfs

git lfs install
git lfs pull

cp .env.example .env
# При необходимости измените пароли (для локальной разработки можно оставить по умолчанию)

docker-compose up -d

# Health check
curl -k https://localhost/health

# Отправить текст на классификацию
curl -k -X POST https://localhost/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This paper presents a new method for optimizing neural networks..."}'

# Открыть веб-интерфейс
open https://localhost
```
