# This Dockerfile is used to deploy a simple single-container Reflex app instance.
FROM python:3.11

# RUN apt-get update && \
#     apt-get install -y redis-server && \
#     rm -rf /var/lib/apt/lists/*

# ENV REFLEX_REDIS_URL=redis://localhost
ENV PYTHONUNBUFFERED=1

# Copy local context to `/app` inside container (see .dockerignore)
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install app requirements and reflex in the container
RUN pip install -r requirements.txt

# Copy the rest of the app files
COPY . .

# Deploy templates and prepare app
RUN reflex init

# Download all npm dependencies and compile frontend
RUN reflex export --frontend-only --no-zip

# Needed until Reflex properly passes SIGTERM on backend.
STOPSIGNAL SIGKILL

# CMD ["sh", "-c", "redis-server --daemonize yes && exec reflex run --env prod"]
CMD ["sh", "-c", "exec reflex run --env prod"]
