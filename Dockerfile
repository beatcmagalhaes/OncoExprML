FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY input/ ./input/

RUN mkdir -p output/tables output/figures

CMD ["python", "src/run_pipeline.py"]