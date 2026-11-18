# syntax = docker/dockerfile:1.4

# Full base: the module is an HTTP client of a llama-server and does no local
# inference, but it imports `openai`, which only requirements.base.txt carries.
ARG base_image=ghcr.io/leolani/cltl-base:latest
FROM ${base_image}

LABEL org.opencontainers.image.source="https://github.com/leolani/cltl-containers"
LABEL org.opencontainers.image.description="Leolani VLM Object Recognition"
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=leolani . /leolani/

WORKDIR /cltl-vlm-object
COPY setup.py requirements.txt README.md VERSION ./
COPY src ./src
# Standalone defaults; a deployment mounts its own config/ over these.
COPY config/default.config config/logging.config ./config/

RUN pip install --no-index --no-build-isolation --find-links=/leolani -r requirements.txt && \
    rm -rf /leolani && \
    find /usr/local/lib/python3.10 -type d -name __pycache__ -exec rm -rf {} +

# Probed with the interpreter rather than curl, as in the slim components.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request\ntry:\n    sys.exit(0 if urllib.request.urlopen('http://localhost:8010/health', timeout=4).status == 200 else 1)\nexcept Exception as e:\n    print(e)\n    sys.exit(1)"]

CMD ["python", "src/main.py"]
