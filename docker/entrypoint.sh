#!/bin/sh
set -eu

if [ "${RUN_MIGRATIONS:-0}" = "1" ]; then
  # Fail fast with a clear message when migration credentials are not provided.
  for required_var in DB_USER DB_PASSWORD DB_NAME; do
    eval "required_value=\${$required_var:-}"
    if [ -z "$required_value" ]; then
      echo "[entrypoint] Missing required env var: $required_var" >&2
      exit 1
    fi
  done

  echo "[entrypoint] Waiting for DB at ${DB_HOST:-db}:${DB_PORT:-3306}"
  python - <<'PY'
import os
import socket
import time

host = os.getenv("DB_HOST", "db")
port = int(os.getenv("DB_PORT", "3306"))
timeout = int(os.getenv("DB_WAIT_TIMEOUT", "60"))

deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"DB at {host}:{port} not reachable after {timeout}s")
PY

  echo "[entrypoint] Running Alembic migrations"
  alembic upgrade head
fi

exec "$@"
