# FRIDA Triton Encoder (ONNX Optimized)

Микросервис для получения эмбеддингов текста: модель **ai-forever/FRIDA** оптимизирована (**FP16**) и развёрнута на **Triton Inference Server (ONNX Runtime)** с поддержкой **динамического батчинга**. Обёртка на FastAPI принимает текст и возвращает векторы размерности **1536**. Система оптимизирована для GPU без тензорных ядер (например, Tesla P40), достигая высокой пропускной способности.

## Основные особенности

- **Triton Server** — инференс модели в ONNX Runtime (v24.01). Используются веса в **FP16** для максимальной скорости на GPU и динамический батчинг.
- **FastAPI Service** — асинхронный сервис, использующий `aiohttp`-клиент к Triton. Поддерживает пакетную обработку и возвращает нормализованные векторы.
- **Python SDK** — встроенный пакет `client` с классом `APIEncoder` для простой интеграции в любой Python-кода.
- **Автоматизация** — `./run.sh` автоматически создаёт виртуальное окружение, экспортирует модель в оптимизированный ONNX (IR v9) и поднимает весь стек.

---

## Структура проекта

```text
.
├── README.md                   # Эта справка
├── run.sh                      # Единая точка запуска: экспорт → docker compose
├── requirements.txt            # Зависимости для экспорта в ONNX
├── docker-compose.yml          # Triton Server + Encoder Service
├── locustfile.py               # Нагрузочное тестирование
├── test.py                     # Тест сравнения (Local Transformers vs Triton)
│
├── client/                     # Python-клиент для Encoder Service
│   ├── api_encoder.py          # Класс APIEncoder
│   └── requirements.txt        # aiohttp, numpy
│
├── service/                    # Код FastAPI сервиса
│   ├── main.py                 # API эндпоинты
│   ├── triton_backend.py       # Логика взаимодействия с Triton
│   └── Dockerfile              # Образ сервиса
│
├── scripts/                    # Скрипты для экспорта
│   ├── export_model.py         # Точка входа для экспорта
│   ├── export_onnx.py          # Конвертация FRIDA (T5) -> ONNX FP16
│   └── model_wrapper.py        # CLS-pooling и L2-нормализация
│
└── models/                     # Репозиторий моделей Triton
    └── frida_model/            # Конфигурация и веса модели
        └── config.pbtxt        # Triton Model Configuration
```

---

## Запуск и использование

### 1. Быстрый старт
```bash
chmod +x run.sh
./run.sh
```
Скрипт сам экспортирует модель (если её нет) и запустит контейнеры.

### 2. Проверка API

**Здоровье сервиса:**
```bash
curl http://localhost:8080/health
```

**Размерность векторов:**
```bash
curl http://localhost:8080/get_vector_dim
```

**Получение эмбеддингов:**
```bash
# Один текст
curl -X POST http://localhost:8080/encode \
     -H "Content-Type: application/json" \
     -d '{"query": "Пример текста для FRIDA"}'

# Пакет (batch)
curl -X POST http://localhost:8080/encode \
     -H "Content-Type: application/json" \
     -d '{"query": ["текст 1", "текст 2"], "prefix": "search_query: "}'
```

---

## Тестирование и отладка

### Локальное тестирование (сравнение)
Файл `test.py` позволяет сравнить результаты инференса через Triton с результатами `SentenceTransformer` (локально):
```bash
python test.py
```

### Нагрузочное тестирование (Locust)
```bash
pip install locust
locust -f locustfile.py --host http://localhost:8080 --headless -u 50 -r 10 -t 60s
```

---

## Системные требования

- ОС: Linux (Ubuntu 20.04+ рекомендуется)
- Docker & NVIDIA Container Toolkit
- GPU: NVIDIA (минимум 8 ГБ VRAM, протестировано на Tesla P40)
- Python 3.12+ (только для запуска `run.sh` — первичный экспорт)
