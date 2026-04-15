FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY run_all_experiments.sh ./run_all_experiments.sh
COPY README.md ./README.md

RUN chmod +x /app/run_all_experiments.sh && mkdir -p /app/results

ENTRYPOINT ["python", "-m", "src.run_experiment"]
CMD ["configs/adult.yaml", "--output-dir", "results/experiments"]
