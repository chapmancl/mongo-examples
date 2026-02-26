FROM node:20-slim AS frontend-build

WORKDIR /build/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir gunicorn

COPY backend/ backend/
COPY --from=frontend-build /build/frontend/dist frontend/dist

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8000", "backend.app:app", "--timeout", "300"]
