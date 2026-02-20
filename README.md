# BGE-M3 Triton Inference Server

Развёртывание модели `bge-m3-legal-ru-updata` для юридических текстов на Triton Server с оптимизацией TensorRT.

## Быстрый старт

Из корня проекта выполните:

```bash
chmod +x run.sh convert_trt.sh
./run.sh
```

Скрипт **run.sh** по шагам:

1. **Создаёт виртуальное окружение** (`triton_env`) и устанавливает зависимости из `requirements.txt`.
2. **Экспортирует модель в TensorRT** только если в каталоге ещё нет `models/bge_model/1/model.plan`:
   - `python export_model.py` — экспорт в ONNX (только Python, без вызова shell);
   - `./convert_trt.sh` — конвертация ONNX → TensorRT в Docker;
   - удаление артефактов ONNX и виртуального окружения.
3. **Запускает Triton Server** в Docker (при необходимости перезапускает уже запущенный контейнер).

Дальше достаточно запускать только Triton (если модель уже экспортирована):

```bash
./run.sh
```

## Порты сервера

- **8000** — HTTP
- **8001** — gRPC
- **8002** — метрики Prometheus

## Структура проекта

```
.
├── README.md
├── run.sh                 # Итоговый запуск: среда → экспорт (при необходимости) → Triton
├── convert_trt.sh         # Конвертация ONNX → TensorRT (вызывается из run.sh, docker + trtexec)
├── export_model.py        # Точка входа: только экспорт модели в ONNX
├── export_onnx.py         # Экспорт модели в ONNX
├── model_wrapper.py       # Обёртка BGE-M3 для ONNX
├── requirements.txt
└── models/
    └── bge_model/
        ├── 1/
        │   └── model.plan   # TensorRT-модель (создаётся при первом run.sh)
        └── config.pbtxt
```

## Проверка

```bash
curl localhost:8000/v2/models/bge_model
```

## Требования

- Linux, Docker с поддержкой GPU (NVIDIA Container Toolkit)
- NVIDIA GPU и актуальные драйверы
- Python 3.12+ (для экспорта модели)
