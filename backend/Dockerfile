FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --only-binary=:all: torch --index-url https://download.pytorch.org/whl/cpu
RUN python -c "import subprocess,tomllib; subprocess.check_call(['pip','install',*tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']])"
COPY backend/app ./backend/app
COPY ml ./ml
RUN useradd --uid 10001 --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
