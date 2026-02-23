# BGE-M3 Triton Encoder

Микросервис для получения эмбеддингов текста: модель BGE-M3 (юридические тексты) развёрнута на Triton Inference Server (TensorRT), обёртка на FastAPI принимает текст по REST и возвращает векторы размерности 1024. Клиент `APIEncoder` (aiohttp) позволяет вызывать сервис из кода без зависимостей на transformers/torch.

---

## Описание проекта

- **Triton** — инференс модели в TensorRT (один запрос = один вектор 1024).
- **Encoder (FastAPI)** — токенизатор в lifespan, асинхронные запросы в Triton; эндпоинты: `/health`, `/encode`, `/get_vector_dim`.
- **Клиент** — пакет `client` с классом `APIEncoder`: те же три операции по HTTP, минимум зависимостей (aiohttp, numpy, tqdm).
- **Запуск** — `run.sh` при необходимости экспортирует модель в ONNX → TensorRT, затем поднимает Triton и Encoder через Docker Compose.

---

## Структура проекта

```
.
├── README.md
├── .gitignore
├── run.sh                      # Запуск: экспорт модели (если нет) → docker compose
├── requirements.txt            # Зависимости для экспорта в ONNX (torch, transformers)
├── docker-compose.yml          # Сервисы: triton, encoder
├── locustfile.py               # Нагрузочное тестирование Encoder
│
├── client/                     # Клиент к Encoder Service
│   ├── __init__.py             # from client import APIEncoder
│   ├── api_encoder.py          # Класс APIEncoder (health, encode, get_vector_dim)
│   └── requirements.txt        # aiohttp, numpy, tqdm — только для клиента
│
├── service/                    # Микросервис FastAPI
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── main.py                 # Эндпоинты /health, /encode, /get_vector_dim
│   ├── triton_backend.py       # Асинхронный aiohttp-клиент к Triton
│   └── requirements.txt       # fastapi, uvicorn, transformers, aiohttp, pydantic
│
├── scripts/                    # Экспорт модели
│   ├── export_model.py         # Точка входа: экспорт в ONNX
│   ├── export_onnx.py          # BGE-M3 → ONNX
│   ├── model_wrapper.py        # Обёртка модели для ONNX (pooling + normalize)
│   └── convert_trt.sh          # ONNX → TensorRT (Docker)
│
└── models/                     # Репозиторий моделей Triton
    └── bge_model/
        ├── config.pbtxt        # Конфиг модели (input 512, output 1024)
        └── 1/
            └── model.plan      # TensorRT-модель (создаётся при первом run.sh)
```

---

## Запуск и тестирование

**Запуск сервиса (из корня):**
```bash
chmod +x run.sh scripts/convert_trt.sh
./run.sh
```

**Проверка Encoder (curl):**
```bash
curl http://localhost:8080/health
curl http://localhost:8080/get_vector_dim
curl -X POST http://localhost:8080/encode -H "Content-Type: application/json" -d '{"query": "тест", "batch_size": 1}'
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
    print((await e.encode('один текст')).shape)
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
