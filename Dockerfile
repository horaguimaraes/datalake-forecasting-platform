FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FUELLAKE_ROOT=/lake \
    MPLBACKEND=Agg

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# The bundled sample lets the container work offline; real data is loaded with `fuellake run`.
COPY data/sample ./data/sample

RUN useradd --create-home app && mkdir -p /lake && chown -R app:app /lake /app
USER app
VOLUME /lake
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# First start builds the demo lake from the sample (about a minute), then serves it.
CMD ["sh", "-c", "[ -f /lake/models/latest.json ] || fuellake demo --root /lake; exec fuellake serve --root /lake --host 0.0.0.0 --port 8000"]
