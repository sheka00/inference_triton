# BGE-M3 Triton Inference Server

Развёртывание модели `bge-m3-legal-ru-updata` для юридических текстов на Triton Server с оптимизацией TensorRT.

## Быстрый старт

Из корня проекта выполните:

```bash
chmod +x run.sh scripts/convert_trt.sh
./run.sh
```

Скрипт **run.sh** по шагам:

1. **Создаёт виртуальное окружение** (`triton_env`) и устанавливает зависимости из `requirements.txt`.
2. **Экспортирует модель в TensorRT** только если в каталоге ещё нет `models/bge_model/1/model.plan`:
   - `python scripts/export_model.py` — экспорт в ONNX (только Python, без вызова shell);
   - `./scripts/convert_trt.sh` — конвертация ONNX → TensorRT в Docker;
   - удаление артефактов ONNX и виртуального окружения.
3. **Запускает весь стек через Docker Compose**:
   - **Triton Server** — сервер инференса моделей
   - **Encoder Service** — FastAPI микросервис для кодирования текстов

После запуска доступны:
- **Triton**: `http://localhost:8000` (HTTP), `http://localhost:8001` (gRPC), `http://localhost:8002` (метрики)
- **Encoder**: `http://localhost:8080` (REST API)

## Структура проекта

```
.
├── README.md
├── run.sh                      # Итоговый запуск: экспорт модели → запуск стека
├── docker-compose.yml          # Triton + Encoder Service
├── requirements.txt             # Зависимости для экспорта модели
├── locustfile.py               # Нагрузочное тестирование
│
├── service/                    # Микросервис FastAPI: текст → Triton → вектор
│   ├── Dockerfile
│   ├── main.py                 # FastAPI приложение
│   ├── triton_backend.py       # Внутренний клиент к Triton (для сервиса)
│   └── requirements.txt
│
├── client/                     # Клиент для работы с Encoder Service
│   ├── __init__.py
│   ├── api_encoder.py          # APIEncoder класс
│   └── requirements.txt       # aiohttp, numpy, tqdm
│
├── scripts/                    # Скрипты экспорта и конвертации модели
│   ├── export_model.py         # Точка входа: экспорт в ONNX
│   ├── export_onnx.py          # Экспорт модели в ONNX
│   ├── model_wrapper.py        # Обёртка BGE-M3 для ONNX
│   └── convert_trt.sh          # Конвертация ONNX → TensorRT
│
├── examples/                   # Примеры использования
│   └── example_usage.py        # Пример работы с APIEncoder
│
└── models/                     # Модели Triton
    └── bge_model/
        ├── config.pbtxt
        └── 1/
            └── model.plan      # TensorRT-модель (создаётся при первом run.sh)
```

## Проверка Triton

```bash
curl localhost:8000/v2/models/bge_model
```

---

## Использование сервиса

Микросервис-обёртка принимает **текст по REST**, сам токенизирует и ходит в Triton. Клиенту не нужны transformers/torch — только HTTP и ваш клиент (например `APIEncoder` на aiohttp).

### Эндпоинты обёртки

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Health check |
| POST | `/encode` | Тело: `{"query": "текст"}` или `{"query": ["a", "b"], "prefix": "query:", "batch_size": 8}` → массив векторов (по 1024 float) |
| GET | `/encode/get_embedding_dimension` | Размерность вектора: `1024` |
| GET | `/get_vector_dim` | То же: `1024` |

### Пример запроса

```bash
curl -X POST http://localhost:8080/encode \
  -H "Content-Type: application/json" \
  -d '{"query": "Договор поставки товара.", "batch_size": 1}'
```

Полная реализация API см. в `service/main.py`.

### Нагрузочный тест (Locust)

Конфигурация тестов: `locustfile.py`

**Базовый тест:**
```bash
pip install locust
locust -f locustfile.py --host http://localhost:8080 --headless -u 50 -r 10 -t 60s
```

**Жёсткое нагрузочное тестирование:**
```bash
locust -f locustfile.py --host http://localhost:8080 --headless -u 200 -r 30 -t 3m
```

---

## Использование клиента

### Установка

```bash
pip install -r client/requirements.txt
```

### Пример использования

См. `examples/example_usage.py` — полный пример работы с APIEncoder.

---

## Требования

- Linux, Docker с поддержкой GPU (NVIDIA Container Toolkit)
- NVIDIA GPU и актуальные драйверы
- Python 3.12+ (для экспорта модели)
