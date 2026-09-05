FROM python:3.12-slim

# CPU-only API image. No CUDA base. First run downloads ~1 GB of Qwen into HF_HOME.
WORKDIR /app
ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    FLUX_DEVICE=cpu \
    FLUX_DTYPE=fp32 \
    FLUX_SERVE_ENGINE=continuous \
    FLUX_LOAD_MODEL=true

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY benchmarks/last_run.json /app/benchmarks/last_run.json

RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000
CMD ["uvicorn", "flux.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
