FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["streamlit", "run", "app.py",      "--server.address=0.0.0.0",      "--server.port=7860",      "--server.headless=true",      "--browser.gatherUsageStats=false"]
