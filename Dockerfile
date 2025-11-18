
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files for uv
COPY .python-version pyproject.toml uv.lock ./

# Install dependencies from uv.lock
RUN uv sync --frozen

# Copy all project files
COPY . .

EXPOSE 8000

# Use uv to run uvicorn
CMD ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]