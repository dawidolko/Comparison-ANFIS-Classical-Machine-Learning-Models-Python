# Streamlit-owy dashboard porownujacy ANFIS z klasycznymi modelami ML.
# Obraz jest samowystarczalny: instaluje zaleznosci i serwuje aplikacje na 8501.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Biblioteki systemowe wymagane przez matplotlib i backendy numeryczne
# (TensorFlow linkuje libgomp). Instalowane przed kodem, zeby warstwa
# z zaleznosciami byla cache'owana miedzy budowaniami.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Uruchamianie jako uzytkownik bez uprawnien roota.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
