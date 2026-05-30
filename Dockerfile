FROM python:3.11-slim

# Java is required by Spark
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# scripts and data are mounted at runtime via docker-compose volumes
# so edits to scripts don't require a rebuild

COPY run_pipeline.sh .
RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
