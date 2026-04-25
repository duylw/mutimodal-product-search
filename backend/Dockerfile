FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn qdrant-client httpx requests python-multipart pydantic-settings sqlalchemy psycopg2-binary

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
