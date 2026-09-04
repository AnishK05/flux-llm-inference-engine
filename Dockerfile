FROM python:3.12-slim

# CPU-only image. Phase 11 will flesh this out; Phase 0 ships the stub.
WORKDIR /app
ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    FLUX_DEVICE=cpu \
    FLUX_DTYPE=fp32

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000
CMD ["uvicorn", "flux.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
