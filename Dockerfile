FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
RUN uv sync --frozen --no-cache

EXPOSE 8000

# Run the application.
CMD ["uv", "run", "python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
