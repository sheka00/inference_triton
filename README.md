### Шпаргалка: запуск и проверка

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
from client.api_encoder import APIEncoder
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