FROM python:3.13

# If the service expects a different port, provide it here (f.e Render expects port 10000)
ARG PORT=8080
# Only set for local/direct access. When TLS is used, the API_URL is assumed to be the same as the frontend.
ARG API_URL
ENV PORT=$PORT
ENV REFLEX_API_URL=${API_URL:-http://localhost:$PORT}
ENV PYTHONUNBUFFERED=1

# Install Caddy and redis server inside image
RUN apt-get update -y && \
    apt-get install -y caddy && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install app requirements and reflex in the container
COPY reflex_apps/requirements.txt .
RUN pip install -r requirements.txt

# Copy local context to `/app` inside container (see .dockerignore)
COPY reflex_apps .
# COPY . .

# Deploy templates and prepare app
RUN reflex init

# Download all npm dependencies and compile frontend
RUN reflex export --frontend-only --no-zip && \
    mv .web/build/client/* /srv/ && \
    rm -rf .web

# Needed until Reflex properly passes SIGTERM on backend.
STOPSIGNAL SIGKILL

EXPOSE $PORT

# Apply migrations before starting the backend.
CMD ["sh", "-c", "caddy start && exec reflex run --env prod --backend-only"]
