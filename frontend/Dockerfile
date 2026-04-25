FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests pillow

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
