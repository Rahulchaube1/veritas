FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY --chmod=755 configs ./configs
COPY --chmod=644 datasets/public_examples.jsonl ./public_examples.jsonl
RUN --mount=type=secret,id=build_ca if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; pip install --no-cache-dir -c configs/constraints.txt . && useradd --create-home --uid 10001 veritas && mkdir -p /app/.cache && chown veritas:veritas /app/.cache
USER veritas
RUN veritas index public_examples.jsonl
EXPOSE 8000
ENV VERITAS_ENV=production
CMD ["veritas","serve","--host","0.0.0.0","--port","8000"]
