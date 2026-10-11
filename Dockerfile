FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    SEGMENTATION_WEIGHTS=weights/unet_final.pth \
    LANDMARK_WEIGHTS=weights/best_landmark_model.pth \
    REPORT_DIR=reports

WORKDIR /app

# CPU-only PyTorch keeps the image small (the CUDA build is several GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# App dependencies (notebook-only packages are skipped)
COPY requirements.txt .
RUN grep -v -E '^(SimpleITK|ipywidgets|jupyter)' requirements.txt > requirements-app.txt \
    && pip install --no-cache-dir -r requirements-app.txt

# Code, weights and assets
COPY app.py pipeline.py risk_profile.json distance_image.jpeg ./
COPY weights/ weights/
COPY static/ static/

RUN mkdir -p reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
