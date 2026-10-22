FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY wescalp ./wescalp
COPY db ./db
# Editável: o código continua em /app, junto do db/schema.sql que ele lê.
RUN pip install --no-cache-dir -e .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "wescalp.collector"]
