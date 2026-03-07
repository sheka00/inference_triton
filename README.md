# BGE-M3 Triton Encoder (ONNX Optimized)

Микросервис для получения эмбеддингов текста: модель BGE-M3 (юридические тексты) оптимизирована (**FP16**) и развёрнута на **Triton Inference Server (ONNX Runtime)** с поддержкой **динамического батчинга**. Обёртка на FastAPI принимает текст и возвращает векторы размерности 1024. Система оптимизирована для GPU без тензорных ядер (например, Tesla P40), достигая пропускной способности ~30 текстов/сек.

---

## Описание проекта

- **Triton** — инференс модели в ONNX Runtime. Используются веса в FP16 для максимальной скорости на GPU и динамический батчинг для утилизации ресурсов.
- **Encoder (FastAPI)** — токенизатор в lifespan, эффективный клиент к Triton с поддержкой **Client-Side Batching** (отправка всего батча в одном запросе).
- **Клиент** — пакет `client` с классом `APIEncoder`: простая обертка для взаимодействия с сервисом.
- **Запуск** — `run.sh` автоматически экспортирует модель в оптимизированный ONNX и поднимает стек через Docker Compose.

---

## Структура проекта

```
.
├── README.md
├── .gitignore
├── run.sh                      # Запуск: экспорт модели (если нет) → docker compose
├── requirements.txt            # Зависимости для экспорта в ONNX (torch, transformers)
├── docker-compose.yml          # Сервисы: triton, encoder
├── locustfile.py               # Нагрузочное тестирование (поддержка батчей 10+)
│
├── client/                     # Клиент к Encoder Service
│   ├── __init__.py             # from client import APIEncoder
│   ├── api_encoder.py          # Класс APIEncoder (health, encode, get_vector_dim)
│   └── requirements.txt        # aiohttp, numpy — только для клиента
│
├── service/                    # Микросервис FastAPI
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── main.py                 # Эндпоинты /health, /encode, /get_vector_dim
│   ├── triton_backend.py       # Асинхронный клиент к Triton (Batch support)
│   └── requirements.txt       # fastapi, uvicorn, transformers, aiohttp, pydantic
│
├── scripts/                    # Экспорт модели
│   ├── export_model.py         # Точка входа: экспорт в ONNX
│   ├── export_onnx.py          # BGE-M3 → ONNX (FP16 weights, FP32 output)
│   └── model_wrapper.py        # Обёртка модели для ONNX (pooling + normalize)
│
└── models/                     # Репозиторий моделей Triton
    └── bge_model/
        ├── config.pbtxt        # Конфиг: Dynamic Batching, Instance Groups
        └── 1/
            └── model.onnx      # Оптимизированная ONNX-модель (~1.1GB)
```

---

## Запуск и тестирование

**Запуск сервиса (из корня):**
```bash
chmod +x run.sh
./run.sh
```

**Проверка Encoder (curl):**
```bash
curl http://localhost:8080/health
curl http://localhost:8080/get_vector_dim
curl -X POST http://localhost:8080/encode -H "Content-Type: application/json" -d '{"query": "тест"}'
# Или несколько текстов:
curl -X POST http://localhost:8080/encode -H "Content-Type: application/json" -d '{"query": ["текст 1", "текст 2"], "batch_size": 32}'
```

**Проверка Triton:**
```bash
curl http://localhost:8000/v2/models/bge_model
curl http://localhost:8002/metrics  # Метрики Prometheus
```

**Проверка Encoder (Python, из корня):**
```bash
pip install -r client/requirements.txt
python -c "
import asyncio
from client import APIEncoder
async def run():
    e = APIEncoder('http://localhost:8080')
    print(await e.health(), await e.get_vector_dim())
    # Один текст
    print((await e.encode('один текст')).shape)
    # Несколько текстов
    print((await e.encode(['текст 1', 'текст 2'])).shape)
    await e.close()
asyncio.run(run())
"
```

**Нагрузка Locust по FastAPI:**
```bash
pip install locust
locust -f locustfile.py --host http://localhost:8080 --headless -u 50 -r 10 -t 60s
```

---

## Требования

- Linux, Docker с поддержкой GPU (NVIDIA Container Toolkit)
- NVIDIA GPU и актуальные драйверы
- Python 3.12+ (для экспорта модели)
