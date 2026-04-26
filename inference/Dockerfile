FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn transformers pillow python-multipart torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --extra-index-url https://download.pytorch.org/whl/cu126

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
