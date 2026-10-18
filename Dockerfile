FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-версия torch заметно меньше стандартной
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

ENV EASYOCR_DIR=/app/.EasyOCR
# веса OCR скачиваем на этапе сборки, чтобы приложение стартовало быстро
RUN python -c "import easyocr; easyocr.Reader(['ru','en'], gpu=False, \
model_storage_directory='/app/.EasyOCR')"

COPY . .
RUN python -m osi.classifier && python make_samples.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", \
     "--server.address=0.0.0.0", "--server.headless=true"]
