FROM --platform=linux/arm64 python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/* && python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-backend.txt .
RUN pip install -r requirements-backend.txt
COPY . .

FROM --platform=arm64 python:3.11-slim
WORKDIR /app
RUN useradd -m newuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY --chown=newuser:newuser . .
USER newuser
EXPOSE 8000
CMD ["uvicorn","src.main:app", "--host", "0.0.0.0", "--port", "8000"]