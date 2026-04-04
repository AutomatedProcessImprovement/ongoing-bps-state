FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build deps are needed for packages such as lxml/scipy when wheels are unavailable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# requirements.txt references a local editable Prosimos path (-e ../prosimos).
# For container builds, install the rest and then install Prosimos from a pip spec.
ARG PROSIMOS_PIP_SPEC="prosimos==2.0.6"
RUN grep -v "^-e ../prosimos" /tmp/requirements.txt > /tmp/requirements.docker.txt \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.docker.txt \
    && pip install "${PROSIMOS_PIP_SPEC}"

COPY . /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "main_web_app:app", "--host", "0.0.0.0", "--port", "8000"]
