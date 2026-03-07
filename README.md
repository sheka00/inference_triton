# FRIDA Triton Encoder (ONNX Optimized)

Микросервис для получения эмбеддингов текста: модель **ai-forever/FRIDA** оптимизирована (**FP16**) и развёрнута на **Triton Inference Server (ONNX Runtime)** с поддержкой **динамического батчинга**. Обёртка на FastAPI принимает текст и возвращает векторы размерности **1536**. Система оптимизирована для GPU без тензорных ядер (например, Tesla P40), достигая высокой пропускной способности.

## Описание проекта

- **Triton** — инференс модели в ONNX Runtime (v24.12). Используются веса в **FP16** для максимальной скорости на GPU и динамический батчинг.
- **Encoder (FastAPI)** — асинхронный сервис на FastAPI, использующий `aiohttp` клиент к Triton с поддержкой пакетной обработки.
- **Клиент** — пакет `client` с классом `APIEncoder`: простая обертка для взаимодействия с сервисом из любого Python-кода.
- **Запуск** — `./run.sh` автоматически экспортирует модель в оптимизированный ONNX (IR v9) и поднимает стек через Docker Compose.

---

## Структура проекта

```
.
├── README.md
├── run.sh                      # Единая точка запуска: экспорт → docker compose
├── requirements.txt            # Зависимости для экспорта в ONNX (torch, transformers)
├── docker-compose.yml          # Стек: Triton Server + Encoder Service
├── locustfile.py               # Код нагрузочного тестирования
│
├── client/                     # Python-клиент для Encoder Service
│   ├── api_encoder.py          # Класс APIEncoder
│   └── requirements.txt        # aiohttp, numpy — только для клиента
│
├── service/                    # Код FastAPI сервиса
│   ├── main.py                 # API эндпоинты
│   └── triton_backend.py       # Логика взаимодействия с Triton
│
├── scripts/                    # Скрипты для экспорта
│   ├── export_model.py         # Запуск процесса экспорта
│   ├── export_onnx.py          # Логика FRIDA -> ONNX FP16
│   └── model_wrapper.py        # CLS-pooling и нормализация вектора
│
└── models/                     # Репозиторий моделей Triton
    └── frida_model/
        ├── config.pbtxt        # Оптимизированный конфиг Triton
        └── 1/
            └── model.onnx      # Экспортированная ONNX-модель
```

---

## Запуск и тестирование

**Запуск всей системы:**
```bash
chmod +x run.sh
./run.sh
```

**Проверка API (curl):**
```bash
# Проверка здоровья
curl http://localhost:8080/health

# Получение векторов
curl -X POST http://localhost:8080/encode \
     -H "Content-Type: application/json" \
     -d '{"query": "Пример текста для FRIDA"}'

# Пакетное получение
curl -X POST http://localhost:8080/encode \
     -H "Content-Type: application/json" \
     -d '{"query": ["текст 1", "текст 2"], "batch_size": 32}'
```

**Проверка Triton:**
```bash
curl http://localhost:8000/v2/models/frida_model
```

**Нагрузочное тестирование (Locust):**
```bash
pip install locust
locust -f locustfile.py --host http://localhost:8080 --headless -u 50 -r 10 -t 60s
```

---

## Требования

- Linux, Docker с поддержкой NVIDIA GPU (NVIDIA Container Toolkit)
- NVIDIA GPU (протестировано на Tesla P40)
- Python 3.12+ (для первичного экспорта модели)
