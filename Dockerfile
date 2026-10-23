FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
WORKDIR /app
COPY requirements-runtime.txt .
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "devbaby.run", "--config", "configs/smoke.json", "--output", "/outputs"]
