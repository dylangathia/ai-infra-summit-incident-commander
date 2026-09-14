FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV IC_PROVIDER=echo
CMD ["sh","-c","uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]